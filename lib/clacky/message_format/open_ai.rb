# frozen_string_literal: true

module Clacky
  module MessageFormat
    # Static helpers for OpenAI-compatible API message format.
    #
    # The canonical internal @messages format IS OpenAI format, so this module
    # mainly handles response parsing, tool result formatting, and message
    # type identification — minimal transformation needed.
    module OpenAI
      module_function

      # ── Message type identification ───────────────────────────────────────────

      # Returns true if the message is a canonical tool result.
      def tool_result_message?(msg)
        msg[:role] == "tool" && !msg[:tool_call_id].nil?
      end

      # Returns the tool_call_ids referenced in a tool result message.
      def tool_call_ids(msg)
        return [] unless tool_result_message?(msg)

        [msg[:tool_call_id]]
      end

      # ── Request building ──────────────────────────────────────────────────────

      # Build an OpenAI-compatible request body.
      #
      # Messages go through the canonical→OpenAI conversion layer
      # (normalize_messages). For most models this is identity because
      # the internal canonical format IS OpenAI format. The conversion
      # handles one edge case: image_url content blocks are stripped
      # when vision_supported is false (e.g. DeepSeek, Kimi, MiniMax),
      # replacing them with a text placeholder so the API doesn't reject
      # the request with "unknown variant 'image_url'".
      #
      # @param messages [Array<Hash>] canonical messages
      # @param model    [String]
      # @param tools    [Array<Hash>] OpenAI-style tool definitions
      # @param max_tokens [Integer]
      # @param caching_enabled [Boolean] (only effective for Claude via OpenRouter)
      # @param vision_supported [Boolean] whether the target model accepts
      #   image_url content blocks (default true, conservative)
      # @return [Hash]
      def build_request_body(messages, model, tools, max_tokens, caching_enabled, vision_supported: true, thinking_level: nil)
        api_messages = messages.map { |msg| normalize_message_content(msg, vision_supported: vision_supported) }

        body = { model: model, max_tokens: max_tokens, messages: api_messages }
        apply_thinking_options!(body, thinking_level)

        if tools&.any?
          if caching_enabled
            cached_tools = deep_clone(tools)
            cached_tools.last[:cache_control] = { type: "ephemeral" }
            body[:tools] = cached_tools
          else
            body[:tools] = tools
          end
        end

        body
      end

      # ── Canonical → OpenAI conversion ─────────────────────────────────────────

      # Process a single message's content through the canonical→OpenAI
      # conversion layer. For String content this is a no-op; for Array
      # content each block goes through normalize_block.
      #
      # @param msg [Hash] canonical message
      # @param vision_supported [Boolean]
      # @return [Hash] message with content normalised for OpenAI API
      def normalize_message_content(msg, vision_supported:)
        content = msg[:content]
        return msg unless content.is_a?(Array)

        blocks = content_to_blocks(content, vision_supported: vision_supported)
        # Most APIs reject empty content arrays — use a placeholder text block.
        blocks = [{ type: "text", text: "..." }] if blocks.empty?
        msg.merge(content: blocks)
      end

      # Convert canonical content array to OpenAI-compatible block array.
      # Each block goes through normalize_block; nil results are compacted.
      #
      # @param content [Array<Hash>] canonical content blocks
      # @param vision_supported [Boolean]
      # @return [Array<Hash>]
      def content_to_blocks(content, vision_supported:)
        content.map { |b| normalize_block(b, vision_supported: vision_supported) }.compact
      end

      # Normalize a single canonical content block to OpenAI API format.
      #
      # Canonical text blocks pass through (with cache_control preserved).
      # image_url blocks are kept for vision-capable models and replaced
      # with a text placeholder for non-vision models (DeepSeek, Kimi, etc.).
      #
      # @param block [Hash] canonical content block
      # @param vision_supported [Boolean]
      # @return [Hash, nil] nil for empty-text blocks (dropped)
      def normalize_block(block, vision_supported:)
        return block unless block.is_a?(Hash)

        case block[:type]
        when "text"
          # Drop empty text blocks — most APIs (Anthropic, DeepSeek, etc.)
          # reject { type: "text", text: "" }.
          text = block[:text]
          return nil if text.nil? || text.empty?

          result = { type: "text", text: text }
          result[:cache_control] = block[:cache_control] if block[:cache_control]
          result
        when "image_url"
          if vision_supported
            block  # Pass through — GPT-4V, Gemini, etc. accept image_url
          else
            # Replace with text placeholder so the API doesn't reject the
            # request. The model will still see the context that an image
            # was present (from file_prompt / system_injected metadata).
            { type: "text", text: "[Image content removed — current model does not support vision input]" }
          end
        else
          block  # Pass through unknown block types (tool_use, tool_result, etc.)
        end
      end

      # Build a streaming request body (same as build_request_body but with
      # stream: true + stream_options for usage).  Used when the provider
      # rejects non-streaming requests (e.g. DeepSeek V4 requires streaming).
      #
      # @param messages [Array<Hash>] canonical messages
      # @param model    [String]
      # @param tools    [Array<Hash>] OpenAI-style tool definitions
      # @param max_tokens [Integer]
      # @param caching_enabled [Boolean]
      # @param vision_supported [Boolean]
      # @return [Hash]
      def build_stream_request_body(messages, model, tools, max_tokens, caching_enabled, vision_supported: true, thinking_level: nil)
        body = build_request_body(messages, model, tools, max_tokens, caching_enabled, vision_supported: vision_supported, thinking_level: thinking_level)
        body[:stream] = true
        # Ask the API to include token usage in the final chunk (OpenAI >= 2024).
        # Older proxies silently ignore unknown fields, so this is safe.
        body[:stream_options] = { include_usage: true }
        body
      end

      # Parse server-sent event (SSE) streaming chunks into canonical format.
      #
      # OpenAI streaming format (NDJSON over SSE):
      #   data: {"choices":[{"delta":{"content":"..."}}]}
      #   data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"..."}}]}}]}
      #   data: {"choices":[{"finish_reason":"stop"}],"usage":{...}}
      #   data: [DONE]
      #
      # Tool calls arrive as incremental fragments across multiple chunks;
      # we accumulate by index and concatenate name/arguments.
      #
      # @param raw_chunks [Array<String>] raw response body chunks from Faraday on_data
      # @return [Hash] canonical response: { content:, tool_calls:, finish_reason:, usage:, raw_api_usage: }
      def parse_stream_response(raw_chunks)
        body = raw_chunks.join
        lines = body.split("\n")

        content = +""
        reasoning_content = +""
        tool_calls_map = {}  # index => { id:, type:, name:, arguments: }
        finish_reason = nil
        usage_data = {}

        idx = 0
        while idx < lines.size
          line = lines[idx]
          idx += 1

          next unless line.start_with?("data: ")

          data_str = line[6..]  # strip "data: " prefix
          next if data_str.strip == "[DONE]"

          chunk = begin
            JSON.parse(data_str)
          rescue JSON::ParserError
            # Faraday's on_data splits by network packet boundary, not by SSE
            # line boundary. A single "data: {...}" line may be split across
            # multiple chunks, producing incomplete JSON on the first line.
            # Try merging subsequent lines until we get valid JSON.
            merged = data_str
            while idx < lines.size
              next_line = lines[idx]
              idx += 1
              # Stop if we hit a new SSE event or DONE marker
              break if next_line.start_with?("data: ") || next_line.strip.empty?
              merged << next_line
              begin
                break chunk = JSON.parse(merged)
              rescue JSON::ParserError
                # Continue merging
              end
            end
            # If we still can't parse, skip this fragment
            next unless chunk.is_a?(Hash)
            chunk
          end

          choice = chunk.dig("choices", 0)

          # Usage arrives in the final chunk where choices is an empty array.
          # Must extract usage BEFORE the "next unless choice" guard.
          usage_data = chunk["usage"] if chunk["usage"].is_a?(Hash) && chunk["usage"]["prompt_tokens"]

          next unless choice

          delta = choice["delta"] || {}

          # Accumulate text content
          content << delta["content"] if delta["content"]

          # Accumulate reasoning content (DeepSeek/Kimi style field).
          # Multiple delta field names exist across vendors:
          #   - reasoning_content  (DeepSeek V4/V3/R1)
          #   - reasoning          (OpenAI-compatible proxies)
          #   - thinking           (alternative proxy convention)
          #   - thought            (some Chinese vendor APIs)
          reasoning_content << delta["reasoning_content"] if delta["reasoning_content"]
          reasoning_content << delta["reasoning"] if delta["reasoning"]
          reasoning_content << delta["thinking"] if delta["thinking"]
          reasoning_content << delta["thought"] if delta["thought"]

          # Accumulate tool calls (arrive incrementally by index)
          if delta["tool_calls"]
            delta["tool_calls"].each do |tc|
              tc_idx = tc["index"]
              tool_calls_map[tc_idx] ||= { id: +"", type: "function", name: +"", arguments: +"" }
              entry = tool_calls_map[tc_idx]
              # id and name are complete values (sent once or repeated identically),
              # NOT incrementally built like arguments — use = not <<.
              # Using << would concatenate the same value across chunks,
              # producing an illegally long call_id (900+ chars) on providers
              # that repeat the id field in every streaming delta.
              entry[:id]        = tc["id"]                       if tc["id"]
              entry[:type]       = tc["type"] || "function"
              entry[:name]      = tc.dig("function", "name")     if tc.dig("function", "name")
              entry[:arguments] << tc.dig("function", "arguments") if tc.dig("function", "arguments")
            end
          end

          # Track finish reason (appears in later chunks)
          finish_reason = choice["finish_reason"] if choice["finish_reason"]

          # Redundant usage extraction: some providers include usage in chunks
          # that also have choices (not just the final empty-choices chunk).
          usage_data = chunk["usage"] if chunk["usage"] && chunk["usage"]["prompt_tokens"]
        end

        # Build canonical tool_calls array sorted by index
        tool_calls = tool_calls_map.keys.sort.map do |k|
          tc = tool_calls_map[k]
          # Convert the accumulated string buffers back to plain strings
          { id: tc[:id].to_s, type: tc[:type], name: tc[:name].to_s, arguments: tc[:arguments].to_s }
        end
        tool_calls = nil if tool_calls.empty?

        usage = {
          prompt_tokens:     usage_data["prompt_tokens"],
          completion_tokens: usage_data["completion_tokens"],
          total_tokens:      usage_data["total_tokens"]
        }
        # Preserve extended usage fields when present
        usage[:api_cost]                    = usage_data["cost"]                            if usage_data["cost"]
        usage[:cache_creation_input_tokens] = usage_data["cache_creation_input_tokens"]     if usage_data["cache_creation_input_tokens"]
        usage[:cache_read_input_tokens]     = usage_data["cache_read_input_tokens"]         if usage_data["cache_read_input_tokens"]
        # OpenRouter stores cache info under prompt_tokens_details
        if (details = usage_data["prompt_tokens_details"])
          usage[:cache_read_input_tokens]     = details["cached_tokens"]    if details["cached_tokens"].to_i > 0
          usage[:cache_creation_input_tokens] = details["cache_write_tokens"] if details["cache_write_tokens"].to_i > 0
        end

        result = {
          content:       content.empty? ? nil : content,
          tool_calls:    tool_calls,
          finish_reason: finish_reason,
          usage:         usage,
          raw_api_usage: usage_data
        }

        # Preserve reasoning_content (e.g. Kimi/Moonshot extended thinking)
        result[:reasoning_content] = reasoning_content unless reasoning_content.empty?

        result
      end

      # ── Response parsing ──────────────────────────────────────────────────────

      # Parse OpenAI-compatible API response into canonical internal format.
      # @param data [Hash] parsed JSON response body
      # @return [Hash]
      def parse_response(data)
        message       = data["choices"].first["message"]
        usage         = data["usage"] || {}
        raw_api_usage = usage.dup

        extracted_thinking = extract_leading_thinking_block(message["content"])
        message_content = extracted_thinking[:content]
        extracted_reasoning = extracted_thinking[:reasoning_content]

        usage_data = {
          prompt_tokens:     usage["prompt_tokens"],
          completion_tokens: usage["completion_tokens"],
          total_tokens:      usage["total_tokens"]
        }

        usage_data[:api_cost]                    = usage["cost"]                            if usage["cost"]
        usage_data[:cache_creation_input_tokens] = usage["cache_creation_input_tokens"]     if usage["cache_creation_input_tokens"]
        usage_data[:cache_read_input_tokens]     = usage["cache_read_input_tokens"]         if usage["cache_read_input_tokens"]

        # OpenRouter stores cache info under prompt_tokens_details
        if (details = usage["prompt_tokens_details"])
          usage_data[:cache_read_input_tokens]     = details["cached_tokens"]    if details["cached_tokens"].to_i > 0
          usage_data[:cache_creation_input_tokens] = details["cache_write_tokens"] if details["cache_write_tokens"].to_i > 0
        end

        result = {
          content:       message_content,
          tool_calls:    parse_tool_calls(message["tool_calls"]),
          finish_reason: data["choices"].first["finish_reason"],
          usage:         usage_data,
          raw_api_usage: raw_api_usage
        }

        # Preserve reasoning_content (DeepSeek / Kimi / OpenAI-compatible thinking).
        # Multiple field names exist across vendors:
        #   - reasoning_content  (DeepSeek V4/V3/R1, some proxies)
        #   - reasoning          (OpenAI-compatible proxies)
        #   - thinking           (alternative proxy convention)
        #   - thought            (some Chinese vendor APIs)
        reasoning = message["reasoning_content"] || message["reasoning"] || message["thinking"] || message["thought"] || extracted_reasoning
        result[:reasoning_content] = reasoning if reasoning

        result
      end

      # ── Tool result formatting ────────────────────────────────────────────────

      # Format tool results into canonical messages to append to @messages.
      # @return [Array<Hash>] canonical tool messages
      def format_tool_results(response, tool_results)
        results_map = tool_results.each_with_object({}) { |r, h| h[r[:id]] = r }

        response[:tool_calls].map do |tc|
          result = results_map[tc[:id]]
          raw_content = result ? result[:content] : { error: "Tool result missing" }.to_json

          # OpenAI tool message content must be a String.
          # If a tool returned multipart Array blocks (e.g. screenshot image), convert to JSON.
          content = raw_content.is_a?(Array) ? JSON.generate(raw_content) : raw_content

          {
            role:         "tool",
            tool_call_id: tc[:id],
            content:      content
          }
        end
      end

      # ── Private helpers ───────────────────────────────────────────────────────

      private_class_method def self.apply_thinking_options!(body, thinking_level)
        level = thinking_level.to_s.strip.downcase
        return body if level.empty?

        body[:reasoning_effort] = level
        body[:reasoning] = { effort: level }
        body
      end

      private_class_method def self.parse_tool_calls(raw)
        return nil if raw.nil? || raw.empty?

        raw.filter_map do |call|
          func = call["function"] || {}
          name = func["name"]
          arguments = func["arguments"]
          # Skip malformed tool calls where name or arguments is nil (broken API response)
          next if name.nil? || arguments.nil?

          { id: call["id"], type: call["type"], name: name, arguments: arguments }
        end
      end

      private_class_method def self.extract_leading_thinking_block(content)
        return { content: content, reasoning_content: nil } unless content.is_a?(String) && !content.empty?

        normalized = content.sub(/\A\s+/, "")

        if normalized.start_with?("<think>")
          open_tag = "<think>"
          close_tag = "</think>"
        elsif normalized.start_with?("<thinking>")
          open_tag = "<thinking>"
          close_tag = "</thinking>"
        else
          return { content: content, reasoning_content: nil }
        end

        close_index = normalized.index(close_tag)
        return { content: content, reasoning_content: nil } unless close_index

        reasoning_content = normalized[open_tag.length...close_index].to_s.strip
        remaining_content = normalized[(close_index + close_tag.length)..].to_s.sub(/\A\s+/, "")

        {
          content: remaining_content,
          reasoning_content: reasoning_content
        }
      end

      private_class_method def self.deep_clone(obj)
        case obj
        when Hash  then obj.each_with_object({}) { |(k, v), h| h[k] = deep_clone(v) }
        when Array then obj.map { |item| deep_clone(item) }
        else obj
        end
      end
    end
  end
end
