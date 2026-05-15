# frozen_string_literal: true

require "json"
require "securerandom"

module Clacky
  module MessageFormat
    # Static helpers for OpenAI Responses API message format.
    #
    # The Responses API (POST /v1/responses) is a newer protocol that replaces
    # Chat Completions.  Key differences:
    #
    #   | Feature          | Chat Completions              | Responses API                 |
    #   |------------------|-------------------------------|-------------------------------|
    #   | Request field    | messages: [...]               | input: [...]                  |
    #   | Token limit      | max_tokens                    | max_output_tokens             |
    #   | Tool calls       | embedded in assistant message | standalone function_call items|
    #   | Tool results     | role: "tool" messages         | function_call_output items    |
    #   | Streaming events | data: {"choices":[...]}       | event: response.output_text.delta |
    #   | Response output  | choices[0].message            | output: [{type: "message"}]   |
    #
    # Some providers (e.g. DeepSeek V4 via OpenRouter) require this protocol
    # and reject Chat Completions requests with "Stream must be set to true".
    #
    # The canonical internal @messages format IS OpenAI Chat Completions format,
    # so this module handles bidirectional conversion.
    module Responses
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

      # Build input items from canonical messages for the Responses API.
      # Used by the OpenAI SDK integration to construct the `input` parameter.
      #
      # @param messages [Array<Hash>] canonical messages
      # @param vision_supported [Boolean] whether the target model accepts images
      # @return [Array<Hash>] Responses API input items
      def build_input_items(messages, vision_supported: true)
        messages.flat_map { |msg| convert_message_to_input_items(msg, vision_supported: vision_supported) }
      end

      # Build a Responses API request body.
      #
      # Converts canonical messages (Chat Completions format) into the
      # Responses API `input` array.  Assistant messages with embedded
      # tool_calls are expanded into separate function_call items.
      #
      # When previous_response_id is provided, the API already knows about
      # the conversation history up to that response — including any
      # function_call items in the previous response's output.  The input
      # array should only contain items that are NEW since that response
      # (e.g. the latest user message and any function_call_output items
      # for tool results).
      #
      # @param messages [Array<Hash>] canonical messages (only new items when
      #   previous_response_id is set)
      # @param model    [String]
      # @param tools    [Array<Hash>] OpenAI-style tool definitions
      # @param max_tokens [Integer] maps to max_output_tokens
      # @param caching_enabled [Boolean] (unused: Responses API has its own caching)
      # @param vision_supported [Boolean] whether the target model accepts images
      # @param previous_response_id [String, nil] previous response ID for
      #   multi-turn continuity
      # @return [Hash]
      def build_request_body(messages, model, tools, max_tokens, caching_enabled, vision_supported: true, previous_response_id: nil, reasoning_effort: nil)
        input_items = messages.flat_map { |msg| convert_message_to_input_items(msg, vision_supported: vision_supported, previous_response_id: previous_response_id) }

        body = {
          model: model,
          input: input_items,
          max_output_tokens: max_tokens,
          store: false
        }
        apply_reasoning_options!(body, reasoning_effort)

        body[:previous_response_id] = previous_response_id if previous_response_id

        if tools&.any?
          body[:tools] = tools.map { |t| convert_tool_to_responses_format(t) }
          body[:tool_choice] = "auto"
        end

        body
      end

      # Build a streaming Responses API request body.
      # Same as build_request_body but with stream: true.
      #
      # @return [Hash]
      def build_stream_request_body(messages, model, tools, max_tokens, caching_enabled, vision_supported: true, previous_response_id: nil, reasoning_effort: nil)
        body = build_request_body(
          messages, model, tools, max_tokens, caching_enabled,
          vision_supported: vision_supported,
          previous_response_id: previous_response_id,
          reasoning_effort: reasoning_effort
        )
        body[:stream] = true
        body
      end

      # ── Canonical → Responses input conversion ────────────────────────────────

      # Convert a single canonical message into one or more Responses API
      # input items.
      #
      #   { role: "system", content: "text" }    → [{ role: "system", content: "text" }]
      #   { role: "user",    content: "text" }    → [{ role: "user",    content: "text" }]
      #   { role: "assistant", content: "text", tool_calls: [...] }
      #     → content part (if present) as { role: "assistant", content: "text" }
      #     → each tool_call as { type: "function_call", id:, call_id:, name:, arguments:, status: }
      #   { role: "tool", tool_call_id:, content: } → { type: "function_call_output", call_id:, output: }
      #
      # When previous_response_id is set, assistant messages are skipped
      # entirely — the API already has them from the previous response.
      #
      # @param msg [Hash] canonical message
      # @param vision_supported [Boolean]
      # @param previous_response_id [String, nil]
      # @return [Array<Hash>]
      private def convert_message_to_input_items(raw_msg, vision_supported:, previous_response_id: nil)
        # Normalize to symbol keys for consistent access — callers may pass
        # either string-keyed or symbol-keyed hashes.
        msg = deep_symbolize_keys(raw_msg)
        role = msg[:role]

        case role
        when "system", "user"
          content = canonicalize_content(msg[:content], vision_supported: vision_supported)
          [{ role: role, content: content }]

        when "assistant"
          # When continuing via previous_response_id, the API already knows
          # about all assistant messages and their function_call items from
          # the previous response's output.  Re-sending them as input causes
          # pairing-constraint violations (e.g. reasoning+message pairing).
          return [] if previous_response_id

          items = []
          # Emit text content as a message item when present
          if msg[:content] && !msg[:content].to_s.empty?
            items << { role: "assistant", content: msg[:content].to_s }
          end
          # Emit each tool call as a function_call item.
          # Tool calls may be stored in either flat format ({name:, arguments:})
          # or wrapped format ({function: {name:, arguments:}}) — handle both.
          #
          # In the Responses API, function_call items need two IDs:
          #   - id:      a unique item identifier (must start with "fc_")
          #   - call_id: the LLM's call identifier (typically "call_xxx")
          # Our canonical format only stores the LLM's call_id as tc[:id],
          # so we generate a fresh fc_ ID for the id field.
          #
          # The status field is required by the API for function_call items
          # in the input array.  Assistant-produced tool calls are always
          # "completed" (the LLM has finished producing the call).
          if msg[:tool_calls].is_a?(Array)
            msg[:tool_calls].each do |tc|
              func = tc[:function] || tc
              name = func[:name] || tc[:name]
              arguments = func[:arguments] || tc[:arguments]

              items << {
                type: "function_call",
                id: "fc_#{SecureRandom.hex(12)}",
                call_id: tc[:id],
                name: name,
                arguments: arguments.to_s,
                status: "completed"
              }
            end
          end
          items

        when "tool"
          [{
            type: "function_call_output",
            call_id: msg[:tool_call_id],
            output: msg[:content].to_s
          }]

        else
          # Unknown roles — pass through as-is
          [raw_msg]
        end
      end

      # Recursively symbolize all keys in a Hash/Array structure.
      private def deep_symbolize_keys(obj)
        case obj
        when Hash  then obj.each_with_object({}) { |(k, v), h| h[k.to_sym] = deep_symbolize_keys(v) }
        when Array then obj.map { |item| deep_symbolize_keys(item) }
        else obj
        end
      end

      # Normalise content for Responses API input.
      # String content passes through.  Array content (multipart blocks)
      # is converted: text blocks are concatenated, image blocks are kept
      # for vision-capable models or replaced with a placeholder.
      #
      # @param content [String, Array, nil]
      # @param vision_supported [Boolean]
      # @return [String]
      private def canonicalize_content(content, vision_supported:)
        case content
        when String then content
        when Array
          blocks = content.map { |b| normalize_block(b, vision_supported: vision_supported) }.compact
          blocks.empty? ? "" : blocks.map { |b| b[:text] || b["text"] || "" }.join("\n")
        else
          content.to_s
        end
      end

      # Normalise a single content block for Responses API.
      # Text blocks pass through; image blocks are kept for vision models,
      # replaced with a placeholder for text-only models.
      private def normalize_block(block, vision_supported:)
        return block unless block.is_a?(Hash)

        case block[:type] || block["type"]
        when "text"
          text = block[:text] || block["text"]
          return nil if text.nil? || text.empty?
          block
        when "image_url"
          if vision_supported
            block
          else
            { type: "text", text: "[Image content removed — current model does not support vision input]" }
          end
        else
          block
        end
      end

      # ── Non-streaming response parsing ────────────────────────────────────────

      # Parse a non-streaming Responses API response into canonical format.
      #
      # Response shape:
      #   {
      #     "id": "resp_...",
      #     "output": [
      #       { "type": "message", "role": "assistant",
      #         "content": [{ "type": "output_text", "text": "..." }] },
      #       { "type": "function_call", "id": "fc_...", "call_id": "call_...",
      #         "name": "...", "arguments": "{...}" }
      #     ],
      #     "usage": { "input_tokens": N, "output_tokens": N, "total_tokens": N }
      #   }
      #
      # @param data [Hash] parsed JSON response body
      # @return [Hash] canonical response
      def parse_response(data)
        output_items = data["output"] || []
        usage_data   = data["usage"] || {}

        # Extract text content from message-type items
        content_parts = []
        reasoning_parts = []
        tool_calls    = []

        output_items.each do |item|
          case item["type"]
          when "message"
            role = item["role"]
            next unless role == "assistant"

            (item["content"] || []).each do |part|
              case part["type"]
              when "output_text"
                content_parts << part["text"]
              when "reasoning", "reasoning_text", "summary_text"
                reasoning_parts << part["text"] if part["text"]
              end
            end
          when "function_call"
            tool_calls << {
              id:        item["call_id"] || item["id"],
              type:      "function",
              name:      item["name"],
              arguments: item["arguments"].to_s
            }
          end
        end

        content    = content_parts.empty? ? nil : content_parts.join
        reasoning  = reasoning_parts.empty? ? nil : reasoning_parts.join
        tool_calls = nil if tool_calls.empty?

        usage = {
          prompt_tokens:     usage_data["input_tokens"],
          completion_tokens: usage_data["output_tokens"],
          total_tokens:      usage_data["total_tokens"]
        }

        # Preserve extended usage fields when present
        usage[:api_cost]                    = usage_data["cost"]                       if usage_data["cost"]
        usage[:cache_creation_input_tokens] = usage_data["cache_creation_input_tokens"] if usage_data["cache_creation_input_tokens"]
        usage[:cache_read_input_tokens]     = usage_data["cache_read_input_tokens"]     if usage_data["cache_read_input_tokens"]

        # Responses API doesn't have an explicit finish_reason; infer from tool_calls
        finish_reason = tool_calls && !tool_calls.empty? ? "tool_calls" : "stop"
        result = {
          content:       content,
          tool_calls:    tool_calls,
          finish_reason: finish_reason,
          usage:         usage,
          raw_api_usage: usage_data
        }
        result[:reasoning_content] = reasoning if reasoning
        result
      end

      # ── Streaming response parsing ────────────────────────────────────────────

      # Parse server-sent event (SSE) streaming chunks from Responses API
      # into canonical format.
      #
      # Responses API streaming uses both `event:` and `data:` lines:
      #
      #   event: response.output_text.delta
      #   data: {"delta":"Hello"}
      #
      #   event: response.output_text.delta
      #   data: {"delta":" world"}
      #
      #   event: response.function_call_arguments.done
      #   data: {"arguments":"{...}","name":"lookup_weather","call_id":"call_abc"}
      #
      #   event: response.completed
      #   data: {"response":{"output":[],"usage":{...}}}
      #
      # Key insight: response.completed may have `output: []` — all content
      # arrives through the streaming events, not the final event.
      #
      # The parser is tolerant: if `event:` lines are absent, it falls back
      # to inspecting the data payload's shape to classify the event.
      #
      # @param raw_chunks [Array<String>] raw response body chunks from Faraday on_data
      # @return [Hash] canonical response: { content:, tool_calls:, finish_reason:, usage:, raw_api_usage: }
      def parse_stream_response(raw_chunks)
        body  = raw_chunks.join
        events = parse_sse_events(body)

        content     = +""
        reasoning   = +""
        tool_calls  = []
        usage_data  = {}
        finish_reason = "stop"

        events.each do |event|
          case event[:type]
          when "response.output_text.delta"
            delta = event.dig(:data, "delta")
            content << delta if delta

          when "response.output_text.done"
            # Complete text delivered — we already accumulated via deltas

          when "response.reasoning.delta", "response.reasoning_text.delta"
            delta = event.dig(:data, "delta")
            reasoning << delta if delta

          when "response.reasoning_summary_text.delta"
            delta = event.dig(:data, "delta")
            reasoning << delta if delta

          when "response.function_call_arguments.delta"
            # Arguments arrive incrementally; accumulate into the current tool call.
            # When the last entry already has an id (completed via .done / output_item.done),
            # start a new entry — otherwise deltas for the next call would corrupt it.
            delta = event.dig(:data, "delta")
            if delta
              last_done = tool_calls.any? && !tool_calls.last[:id].to_s.empty?
              tool_calls << { id: +"", type: "function", name: +"", arguments: +"" } if tool_calls.empty? || last_done
              tool_calls.last[:arguments] << delta
            end

          when "response.function_call_arguments.done"
            data = event[:data]
            # The done event carries the complete arguments + item_id, but
            # name/call_id may arrive separately via response.output_item.done.
            # Some providers nest name/call_id inside data["item"].
            item = data["item"] || {}
            name_from = item["name"] || data["name"] || ""
            id_from   = item["call_id"] || item["id"] || data["call_id"] || data["id"] || ""
            args_from = item["arguments"] || data["arguments"] || ""
            # Find the most recent incomplete entry (name/id not yet set)
            existing = tool_calls.reverse.find { |tc| tc[:name].to_s.empty? }
            if existing
              existing[:id]        = id_from
              existing[:name]      = name_from
              existing[:arguments] = args_from
            else
              tool_calls << {
                id:        id_from,
                type:      "function",
                name:      name_from,
                arguments: args_from
              }
            end

          when "response.output_item.done"
            item = event.dig(:data, "item") || {}
            case item["type"]
            when "function_call"
              # Complete function_call item — find the most recent incomplete entry
              existing = tool_calls.reverse.find { |tc| tc[:name].to_s.empty? }
              if existing
                existing[:id]        = item["call_id"] || item["id"] || ""
                existing[:name]      = item["name"] || ""
                existing[:arguments] = item["arguments"] || ""
              else
                tool_calls << {
                  id:        item["call_id"] || item["id"],
                  type:      "function",
                  name:      item["name"],
                  arguments: item["arguments"].to_s
                }
              end
            when "message"
              # Complete message item with full content — replace accumulated text
              text = extract_text_from_message_item(item)
              content.replace(text) unless text.empty?
            end

          when "response.completed"
            # Final event — carries usage and optional output
            resp = event.dig(:data, "response") || {}
            usage_data = resp["usage"] || {}
            # Some providers set finish_reason here
            finish_reason = resp["status"] == "completed" ? "stop" : (resp["status"] || "stop")

          when "response.failed"
            finish_reason = "error"

          when "response.incomplete"
            finish_reason = "length"
          end
        end

        # Clean up: remove empty/incomplete tool calls
        tool_calls.reject! { |tc| tc[:name].to_s.empty? }
        tool_calls = nil if tool_calls.empty?

        content = nil if content.empty?

        # When tool_calls are present, override finish_reason to "tool_calls"
        # so the agent loop knows to execute them before stopping.
        finish_reason = "tool_calls" if tool_calls && !tool_calls.empty?

        usage = {
          prompt_tokens:     usage_data["input_tokens"],
          completion_tokens: usage_data["output_tokens"],
          total_tokens:      usage_data["total_tokens"]
        }
        usage[:api_cost] = usage_data["cost"] if usage_data["cost"]

        result = {
          content:       content,
          tool_calls:    tool_calls,
          finish_reason: finish_reason,
          usage:         usage,
          raw_api_usage: usage_data
        }

        result[:reasoning_content] = reasoning unless reasoning.empty?

        result
      end

      # ── Tool result formatting ────────────────────────────────────────────────

      # Format tool results into canonical messages for Responses API.
      #
      # In Responses API, tool results are sent as function_call_output items
      # in the input array.  However, the canonical @messages format uses
      # `role: "tool"` messages and the Client converts them when building
      # the request body.  So this method returns standard canonical tool
      # messages — the conversion to function_call_output happens in
      # convert_message_to_input_items when the next request is built.
      #
      # @param response [Hash] canonical response with :tool_calls
      # @param tool_results [Array<Hash>] executed tool results
      # @return [Array<Hash>] canonical tool messages
      def format_tool_results(response, tool_results)
        results_map = tool_results.each_with_object({}) { |r, h| h[r[:id]] = r }

        (response[:tool_calls] || []).map do |tc|
          result      = results_map[tc[:id]]
          raw_content = result ? result[:content] : { error: "Tool result missing" }.to_json

          # Responses API function_call_output expects a string
          content = raw_content.is_a?(Array) ? JSON.generate(raw_content) : raw_content.to_s

          {
            role:         "tool",
            tool_call_id: tc[:id],
            content:      content
          }
        end
      end

      # ── Private helpers ───────────────────────────────────────────────────────

      # Parse raw SSE body into an array of { type:, data: } hashes.
      # Handles both standard (event: ...\ndata: ...) and simple (data: only) formats.
      private def apply_reasoning_options!(body, reasoning_effort)
        level = reasoning_effort.to_s.strip.downcase
        return body if level.empty?

        body[:reasoning] = { effort: level }
        body
      end

      # Parse raw SSE body into an array of { type:, data: } hashes.
      # Handles both standard (event: ...\ndata: ...) and simple (data: only) formats.
      private def parse_sse_events(body)
        events     = []
        current_type = nil
        current_data = nil

        body.each_line do |line|
          line = line.chomp

          if line.start_with?("event: ")
            current_type = line[7..].strip
          elsif line.start_with?("data: ")
            data_str = line[6..]
            next if data_str.strip == "[DONE]"

            parsed = begin
              JSON.parse(data_str)
            rescue JSON::ParserError
              nil
            end
            next unless parsed

            # If no event type was declared, infer it from the data shape
            unless current_type
              current_type = infer_event_type(parsed)
            end

            current_data = parsed
          elsif line.empty?
            # Empty line = end of event.  Flush the current event.
            if current_type && current_data
              events << { type: current_type, data: current_data }
            end
            current_type = nil
            current_data = nil
          end
        end

        # Flush the last event if the stream didn't end with a blank line
        if current_type && current_data
          events << { type: current_type, data: current_data }
        end

        events
      end

      # Infer the event type from the data payload shape when `event:` lines
      # are absent.  Used as a fallback for providers that omit the event field.
      private def infer_event_type(data)
        # Check for known top-level keys
        if data.key?("delta") && !data.key?("response")
          "response.output_text.delta"
        elsif data.key?("text") && !data.key?("response")
          "response.output_text.done"
        elsif data.key?("arguments") && data.key?("name")
          "response.function_call_arguments.done"
        elsif data.key?("item") && data["item"].is_a?(Hash)
          case data.dig("item", "type")
          when "function_call" then "response.output_item.done"
          when "message"       then "response.output_item.done"
          else                      "response.output_item.added"
          end
        elsif data.key?("response")
          "response.completed"
        else
          "unknown"
        end
      end

      # Extract plain text from a Responses API message output item.
      # @param item [Hash] e.g. { "type": "message", "content": [{ "type": "output_text", "text": "..." }] }
      # @return [String]
      private def extract_text_from_message_item(item)
        (item["content"] || [])
          .select { |c| c["type"] == "output_text" }
          .map { |c| c["text"] }
          .join
      end

      # Convert a tool definition from Chat Completions format to Responses API format.
      # Chat Completions: { type: "function", function: { name:, description:, parameters: } }
      # Responses API:    { type: "function", name:, description:, parameters: }
      def convert_tool_to_responses_format(tool)
        # Support both symbol and string keys
        func = tool[:function] || tool["function"] || tool
        func_name = func[:name] || func["name"] || tool[:name] || tool["name"]
        func_desc = func[:description] || func["description"] || tool[:description] || tool["description"]
        func_params = func[:parameters] || func["parameters"] || tool[:parameters] || tool["parameters"]
        {
          type: "function",
          name: func_name,
          description: func_desc,
          parameters: func_params
        }
      end

      private def deep_clone(obj)
        case obj
        when Hash  then obj.each_with_object({}) { |(k, v), h| h[k] = deep_clone(v) }
        when Array then obj.map { |item| deep_clone(item) }
        else obj
        end
      end
    end
  end
end
