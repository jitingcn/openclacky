# frozen_string_literal: true

# logger was removed from Ruby 4.0 stdlib; faraday needs it
require "logger"

require "faraday"
require "json"

module Clacky
  class Client
    MAX_RETRIES = 10
    RETRY_DELAY = 5 # seconds

    def initialize(api_key, base_url:, model:, anthropic_format: false, api_type: nil, stream: nil)
      @api_key = api_key
      @base_url = base_url
      @model = model
      @stream = stream  # true = always stream, false = never stream, nil = auto

      # Responses API state tracking for multi-turn conversation continuity.
      # The Responses API is stateful — it expects previous_response_id to chain
      # turns.  We track both the last response ID and the message count at the
      # time of the last request so we can send only the delta on the next turn.
      #
      # Some proxies (OpenRouter, custom gateways) reject previous_response_id.
      # @responses_supports_prev_id tracks whether the upstream accepts it:
      #   nil  = unknown (will try on next turn)
      #   true = supported
      #   false = rejected — always send full history
      @last_responses_id = nil
      @last_responses_message_count = 0
      @responses_supports_prev_id = nil

      # Resolve provider for capability + anthropic_format lookups (includes
      # clacky-* key fallback for local-debug proxy setups).
      provider_id = Providers.resolve_provider(base_url: @base_url, api_key: @api_key)

      # ── api_type resolution (explicit config > anthropic_format back-compat > base_url match) ──
      # Only auto-detect api_type from provider preset when base_url explicitly
      # matches — do NOT use the clacky-* key fallback (it would incorrectly
      # route a localhost proxy to Bedrock when the user is using Chat Completions).
      resolved_api_type = api_type
      base_url_provider_id = Providers.find_by_base_url(@base_url)

      if resolved_api_type.nil? || resolved_api_type.to_s.strip.empty?
        # Backward compatibility: anthropic_format=true implies api_type="anthropic-messages"
        if anthropic_format
          resolved_api_type = "anthropic-messages"
        else
          # Auto-detect from provider preset (only when base_url matches)
          resolved_api_type = base_url_provider_id ? Providers.api_type(base_url_provider_id) : nil
        end
      end

      # ── Set routing flags from resolved api_type ──
      @use_anthropic_format = false
      @use_bedrock = false
      @use_responses = false

      case resolved_api_type
      when "anthropic-messages"
        @use_anthropic_format = true
      when "bedrock"
        @use_bedrock = true
      when "openai-responses"
        @use_responses = true
      when "openai-completions", nil
        # Chat Completions (default path) — nothing special needed
      end

      # Also run the legacy Bedrock detection as a safety net
      # (ABSK key prefix or abs- model prefix)
      @use_bedrock ||= MessageFormat::Bedrock.bedrock_api_key?(api_key, model)

      # Decide anthropic_format dynamically based on provider+model, falling
      # back to the explicit constructor flag for unknown providers / custom
      # base_urls. This lets e.g. OpenRouter's Claude models auto-route to the
      # native /v1/messages endpoint (preserving cache_control byte-for-byte)
      # without requiring any change to user YAML.
      provider_prefers_anthropic = provider_id &&
                                   Providers.anthropic_format_for_model?(provider_id, @model)
      @use_anthropic_format = provider_prefers_anthropic || anthropic_format

      # Remember the provider id so we can tune connection headers below
      # (OpenRouter's /v1/messages accepts either Bearer or x-api-key, but
      # some OpenRouter-compatible relays only honour Bearer — send both).
      @provider_id = provider_id

      # Determine vision support once at construction time.
      # Non-vision models (DeepSeek, Kimi, MiniMax, etc.) reject image_url
      # content blocks; the conversion layer strips them when this is false.
      @vision_supported = Providers.supports?(provider_id, :vision, model_name: @model)
    end

    # Returns true when the client is using the AWS Bedrock Converse API.
    def bedrock?
      @use_bedrock
    end

    # Returns true when the client is talking directly to the Anthropic API
    # (determined at construction time via the anthropic_format flag).
    def anthropic_format?(model = nil)
      @use_anthropic_format && !@use_bedrock
    end

    # ── Connection test ───────────────────────────────────────────────────────

    # Test API connection by sending a minimal request.
    # Returns { success: true } or { success: false, error: "..." }.
    def test_connection(model:)
      if bedrock?
        body = MessageFormat::Bedrock.build_request_body(
          [{ role: :user, content: "hi" }], model, [], 16
        ).to_json
        response = bedrock_connection.post(bedrock_endpoint(model)) { |r| r.body = body }
      elsif anthropic_format?
        minimal_body = { model: model, max_tokens: 16,
                         messages: [{ role: "user", content: "hi" }] }.to_json
        response = anthropic_connection.post(anthropic_messages_path) { |r| r.body = minimal_body }
      elsif @use_responses
        minimal_body = { model: model, max_output_tokens: 16, store: false,
                         input: [{ role: "user", content: "hi" }] }
        minimal_body[:stream] = true if @stream
        response = responses_connection.post("responses") { |r| r.body = minimal_body.to_json }

        # Auto mode: retry with streaming if server requires it
        if response.status == 400 && @stream.nil?
          error_body = begin
            JSON.parse(response.body)
          rescue
            {}
          end
          error_msg = error_body.is_a?(Hash) ? error_body.dig("error", "message").to_s : ""
          if error_msg.match?(/stream/i)
            minimal_body[:stream] = true
            response = responses_connection.post("responses") { |r| r.body = minimal_body.to_json }
          end
        end
      else
        minimal_body = { model: model, max_tokens: 16,
                         messages: [{ role: "user", content: "hi" }] }
        minimal_body[:stream] = true if @stream
        response = openai_connection.post("chat/completions") { |r| r.body = minimal_body.to_json }

        # Auto mode: retry with streaming if server requires it
        if response.status == 400 && @stream.nil?
          error_body = begin
            JSON.parse(response.body)
          rescue
            {}
          end
          error_msg = error_body.is_a?(Hash) ? error_body.dig("error", "message").to_s : ""
          if error_msg.match?(/stream/i)
            minimal_body[:stream] = true
            response = openai_connection.post("chat/completions") { |r| r.body = minimal_body.to_json }
          end
        end
      end
      handle_test_response(response)
    rescue Faraday::Error => e
      { success: false, error: "Connection error: #{e.message}" }
    rescue => e
      Clacky::Logger.error("[test_connection] #{e.class}: #{e.message}", error: e)
      { success: false, error: e.message }
    end

    # ── Simple (non-agent) helpers ────────────────────────────────────────────

    # Send a single string message and return the reply text.
    def send_message(content, model:, max_tokens:)
      messages = [{ role: "user", content: content }]
      send_messages(messages, model: model, max_tokens: max_tokens)
    end

    # Send a messages array and return the reply text.
    def send_messages(messages, model:, max_tokens:)
      if bedrock?
        body     = MessageFormat::Bedrock.build_request_body(messages, model, [], max_tokens)
        response = bedrock_connection.post(bedrock_endpoint(model)) { |r| r.body = body.to_json }
        parse_simple_bedrock_response(response)
      elsif anthropic_format?
        body     = MessageFormat::Anthropic.build_request_body(messages, model, [], max_tokens, false)
        response = anthropic_connection.post(anthropic_messages_path) { |r| r.body = body.to_json }
        # Fall back to Responses API when Anthropic endpoint requires streaming
        if response.status == 400
          error_body = begin; JSON.parse(response.body); rescue; {}; end
          error_msg = error_body.is_a?(Hash) ? error_body.dig("error", "message").to_s : ""
          if error_msg.match?(/stream/i)
            @use_responses = true
            body2 = { model: model, max_output_tokens: max_tokens, store: false, input: messages }
            resp2 = responses_connection.post("responses") { |r| r.body = body2.to_json }
            return html_response?(resp2) ? parse_simple_responses_stream_response(model, messages, max_tokens) : parse_simple_responses_response(resp2)
          end
        end
        if html_response?(response)
          @use_responses = true
          body2 = { model: model, max_output_tokens: max_tokens, store: false, input: messages }
          resp2 = responses_connection.post("responses") { |r| r.body = body2.to_json }
          return html_response?(resp2) ? parse_simple_responses_stream_response(model, messages, max_tokens) : parse_simple_responses_response(resp2)
        end
        parse_simple_anthropic_response(response)
      elsif @use_responses
        body     = { model: model, max_output_tokens: max_tokens, store: false, input: messages }
        response = responses_connection.post("responses") { |r| r.body = body.to_json }
        if response.status == 400
          error_body = begin; JSON.parse(response.body); rescue; {}; end
          error_msg = error_body.is_a?(Hash) ? error_body.dig("error", "message").to_s : ""
          return parse_simple_responses_stream_response(model, messages, max_tokens) if error_msg.match?(/stream/i)
        end
        # HTML response → streaming fallback
        if html_response?(response)
          return parse_simple_responses_stream_response(model, messages, max_tokens)
        end
        parse_simple_responses_response(response)
      else
        body = { model: model, max_tokens: max_tokens, messages: messages }

        # If user explicitly configured stream: true, skip the non-streaming
        # attempt entirely — go straight to streaming (saves 15s timeout wait).
        if @stream == true
          return parse_simple_openai_stream_response(model, messages, max_tokens)
        end

        # Try non-streaming first with a short timeout (15s).  Some servers
        # hang on non-streaming requests because they're streaming-only —
        # catch the timeout and retry with streaming.
        response = begin
          openai_connection.post("chat/completions") { |r| r.body = body.to_json; r.options.timeout = 15 }
        rescue Faraday::TimeoutError
          raise if @stream == false
          return parse_simple_openai_stream_response(model, messages, max_tokens)
        end

        # Detect providers that return "Stream must be set to true" and
        # retry with the streaming endpoint.
        if response.status == 400
          error_body = begin; JSON.parse(response.body); rescue; {}; end
          error_msg = error_body.is_a?(Hash) ? error_body.dig("error", "message").to_s : ""
          if error_msg.match?(/stream/i)
            raise RetryableError, "Provider requires streaming but stream is forced off" if @stream == false
            return parse_simple_openai_stream_response(model, messages, max_tokens)
          end
        end

        # If the non-streaming response is HTML (streaming-only server),
        # fallback to streaming instead of retrying infinitely.
        if html_response?(response)
          raise RetryableError, "Provider requires streaming but stream is forced off" if @stream == false
          return parse_simple_openai_stream_response(model, messages, max_tokens)
        end

        parse_simple_openai_response(response)
      end
    end

    # ── Agent main path ───────────────────────────────────────────────────────

    # Send messages with tool-calling support.
    # Returns canonical response hash: { content:, tool_calls:, finish_reason:, usage:, latency: }
    #
    # Latency measurement:
    #   The primary path is non-streaming (plain POST, response body read in
    #   one shot).  Providers that require streaming (e.g. DeepSeek V4) are
    #   handled via an automatic fallback to SSE streaming; the result is
    #   accumulated and returned identically.  TTFB (time to response headers)
    #   is not exposed by Faraday's default adapter without extra plumbing.
    #   What we CAN measure cheaply — and what users actually feel — is total
    #   request duration, which for a non-streaming call equals the time from
    #   "hit Enter" to "first token visible" (since we receive everything at
    #   once).
    #
    #   So we record `duration_ms` as the authoritative number and alias it to
    #   `ttft_ms` for downstream consumers (status bar uses ttft_ms as its
    #   signal metric — see docs).  When we add full streaming support (with
    #   incremental UI), this same `ttft_ms` field will start carrying the
    #   *actual* first-token latency without any schema change.
    def send_messages_with_tools(messages, model:, tools:, max_tokens:, enable_caching: false)
      caching_enabled = enable_caching && supports_prompt_caching?(model)
      cloned = deep_clone(messages)

      t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      response =
        if @use_responses
          send_responses_request(cloned, model, tools, max_tokens, caching_enabled)
        elsif bedrock?
          send_bedrock_request(cloned, model, tools, max_tokens, caching_enabled)
        elsif anthropic_format?
          send_anthropic_request(cloned, model, tools, max_tokens, caching_enabled)
        else
          send_openai_request(cloned, model, tools, max_tokens, caching_enabled)
        end
      t1 = Process.clock_gettime(Process::CLOCK_MONOTONIC)

      duration_ms = ((t1 - t0) * 1000).round
      # Throughput is only meaningful with a reasonable output size; below ~10
      # tokens the sample is too small to be informative and the result is
      # wildly high (e.g. 1 token / 50ms → 20 tok/s is meaningless).
      # Canonical usage hashes from message_format/* all use :completion_tokens.
      output_tokens = response[:usage]&.dig(:completion_tokens).to_i
      tps = (output_tokens >= 10 && duration_ms > 0) ? (output_tokens * 1000.0 / duration_ms).round(1) : nil

      response[:latency] = {
        ttft_ms:     duration_ms,      # non-streaming: TTFT == full duration
        duration_ms: duration_ms,
        output_tokens: output_tokens,
        tps:         tps,
        model:       model,
        measured_at: Time.now.to_f,
        streaming:   false              # future flag — true when we migrate
      }
      response
    end

    # Format tool results into canonical messages ready to append to @messages.
    # Always returns canonical format (role: "tool") regardless of API type —
    # conversion to API-native happens inside each send_*_request.
    def format_tool_results(response, tool_results, model:)
      return [] if tool_results.empty?

      if @use_responses
        MessageFormat::Responses.format_tool_results(response, tool_results)
      elsif bedrock?
        MessageFormat::Bedrock.format_tool_results(response, tool_results)
      elsif anthropic_format?
        MessageFormat::Anthropic.format_tool_results(response, tool_results)
      else
        MessageFormat::OpenAI.format_tool_results(response, tool_results)
      end
    end

    # ── Prompt-caching support ────────────────────────────────────────────────

    # Returns true for Claude models that support prompt caching (gen 3.5+ or gen 4+).
    #
    # Handles both direct model names (e.g. "claude-haiku-4-5") and
    # Clacky AI Bedrock proxy names with "abs-" prefix (e.g. "abs-claude-haiku-4-5").
    #
    # Why only Claude models:
    #   - MiniMax uses automatic server-side caching (no cache_control needed from client)
    #   - Kimi uses a proprietary prompt_cache_key param, not cache_control
    #   - MiMo has no documented caching API
    #   - Only Claude (direct, OpenRouter, or ClackyAI Bedrock proxy) consumes our
    #     cache_control / cachePoint markers
    def supports_prompt_caching?(model)
      # Strip ClackyAI Bedrock proxy prefix before matching
      model_str = model.to_s.downcase.sub(/^abs-/, "")
      return false unless model_str.include?("claude")

      # Match Claude gen 3.5+ (3.5/3.6/3.7…) or gen 4+ in any name format:
      #   claude-3.5-sonnet-...  claude-3-7-sonnet  claude-haiku-4-5  claude-sonnet-4-6
      model_str.match?(/claude(?:-3[-.]?[5-9]|.*-[4-9][-.]|.*-[4-9]$|-[4-9][-.]|-[4-9]$|-sonnet-[34])/)
    end


    # ── Bedrock Converse request / response ───────────────────────────────────

    def send_bedrock_request(messages, model, tools, max_tokens, caching_enabled)
      body     = MessageFormat::Bedrock.build_request_body(messages, model, tools, max_tokens, caching_enabled)
      response = bedrock_connection.post(bedrock_endpoint(model)) { |r| r.body = body.to_json }

      raise_error(response) unless response.status == 200
      check_html_response(response)
      parsed_body = safe_json_parse(response.body, context: "LLM response")
      MessageFormat::Bedrock.parse_response(parsed_body)
    end

    def parse_simple_bedrock_response(response)
      raise_error(response) unless response.status == 200
      data = safe_json_parse(response.body, context: "LLM response")
      (data.dig("output", "message", "content") || [])
        .select { |b| b["text"] }
        .map { |b| b["text"] }
        .join("")
    end

    # ── Anthropic request / response ──────────────────────────────────────────

    def send_anthropic_request(messages, model, tools, max_tokens, caching_enabled)
      # Apply cache_control to the message that marks the cache breakpoint
      messages = apply_message_caching(messages) if caching_enabled

      body     = MessageFormat::Anthropic.build_request_body(messages, model, tools, max_tokens, caching_enabled)
      response = anthropic_connection.post(anthropic_messages_path) { |r| r.body = body.to_json }

      # Some providers require streaming even for Anthropic-format endpoints.
      # When we get a 400 with "stream" in the error, fall back to Responses API.
      if response.status == 400
        error_body = begin
          JSON.parse(response.body)
        rescue
          {}
        end
        error_msg = error_body.is_a?(Hash) ? error_body.dig("error", "message").to_s : ""
        if error_msg.match?(/stream/i)
          @use_responses = true
          return send_responses_request(messages, model, tools, max_tokens, caching_enabled)
        end
      end

      # When Anthropic endpoint returns HTML, it doesn't exist — fall back to Responses API
      if html_response?(response)
        @use_responses = true
        return send_responses_request(messages, model, tools, max_tokens, caching_enabled)
      end

      raise_error(response) unless response.status == 200
      check_html_response(response)
      parsed_body = safe_json_parse(response.body, context: "LLM response")
      MessageFormat::Anthropic.parse_response(parsed_body)
    end

    def parse_simple_anthropic_response(response)
      raise_error(response) unless response.status == 200
      data = safe_json_parse(response.body, context: "LLM response")
      (data["content"] || []).select { |b| b["type"] == "text" }.map { |b| b["text"] }.join("")
    end

    # ── OpenAI request / response ─────────────────────────────────────────────

    def send_openai_request(messages, model, tools, max_tokens, caching_enabled)
      # Apply cache_control markers to messages when caching is enabled.
      # OpenRouter proxies Claude with the same cache_control field convention as Anthropic direct.
      messages = apply_message_caching(messages) if caching_enabled

      # If user explicitly configured stream: true, skip the non-streaming
      # attempt entirely — go straight to streaming (saves 15s timeout wait).
      if @stream == true
        return send_openai_stream_request(messages, model, tools, max_tokens, caching_enabled)
      end

      body     = MessageFormat::OpenAI.build_request_body(
        messages, model, tools, max_tokens, caching_enabled,
        vision_supported: @vision_supported
      )

      # Try non-streaming first with a short timeout (15s).  Some providers
      # hang on non-streaming requests because they're streaming-only —
      # catch the timeout and retry with streaming.
      response = begin
        openai_connection.post("chat/completions") { |r| r.body = body.to_json; r.options.timeout = 15 }
      rescue Faraday::TimeoutError
        # When stream: false (forced), don't fallback — propagate the error
        raise if @stream == false
        return send_openai_stream_request(messages, model, tools, max_tokens, caching_enabled)
      end

      # Detect providers that return "Stream must be set to true" (e.g. DeepSeek
      # V4) and retry with streaming — the accumulated result is identical.
      if response.status == 400
        error_body = begin
          JSON.parse(response.body)
        rescue
          {}
        end
        error_msg = error_body.is_a?(Hash) ? error_body.dig("error", "message").to_s : ""
        if error_msg.match?(/stream/i)
          raise RetryableError, "Provider requires streaming but stream is forced off" if @stream == false
          return send_openai_stream_request(messages, model, tools, max_tokens, caching_enabled)
        end
      end

      raise_error(response) unless response.status == 200

      # If non-streaming returned HTML (streaming-only server),
      # fallback to streaming instead of getting stuck in a retry loop.
      if html_response?(response)
        raise RetryableError, "Provider requires streaming but stream is forced off" if @stream == false
        return send_openai_stream_request(messages, model, tools, max_tokens, caching_enabled)
      end
      
      parsed_body = safe_json_parse(response.body, context: "LLM response")
      MessageFormat::OpenAI.parse_response(parsed_body)
    end

    # Streaming variant of send_openai_request for providers that require
    # stream: true (e.g. DeepSeek V4).  Reads the response body via Faraday's
    # on_data callback, accumulates SSE chunks, and parses them into the same
    # canonical format as the non-streaming path.
    def send_openai_stream_request(messages, model, tools, max_tokens, caching_enabled)
      body = MessageFormat::OpenAI.build_stream_request_body(
        messages, model, tools, max_tokens, caching_enabled,
        vision_supported: @vision_supported
      )

      chunks = []
      response = openai_connection.post("chat/completions") do |req|
        req.body = body.to_json
        req.options.on_data = proc do |chunk, _bytes, _env|
          chunks << chunk if chunk
        end
      end

      raise_error(response, chunks: chunks) unless response.status == 200
      check_html_response(response, chunks: chunks)

      MessageFormat::OpenAI.parse_stream_response(chunks)
    end

    def parse_simple_openai_response(response)
      raise_error(response) unless response.status == 200
      parsed_body = safe_json_parse(response.body, context: "LLM response")
      parsed_body["choices"].first["message"]["content"]
    end

    # Streaming variant of parse_simple_openai_response for providers that
    # require stream: true.  Returns the accumulated text content.
    def parse_simple_openai_stream_response(model, messages, max_tokens)
      body = { model: model, max_tokens: max_tokens, messages: messages,
               stream: true, stream_options: { include_usage: true } }

      chunks = []
      response = openai_connection.post("chat/completions") do |req|
        req.body = body.to_json
        req.options.on_data = proc do |chunk, _bytes, _env|
          chunks << chunk if chunk
        end
      end

      raise_error(response, chunks: chunks) unless response.status == 200
      check_html_response(response, chunks: chunks)

      parsed = MessageFormat::OpenAI.parse_stream_response(chunks)
      parsed[:content] || ""
    end

    # ── Responses API request / response ──────────────────────────────────────

    # Send a tool-calling request via the Responses API (POST /v1/responses).
    # Tries non-streaming first; falls back to streaming on 400 errors that
    # mention "stream" (e.g. DeepSeek V4 requires streaming).
    #
    # Multi-turn continuity: after the first response we track
    # @last_responses_id and only send NEW messages since the last request.
    # The previous_response_id lets the API access the full conversation
    # history without us having to re-send every assistant message and its
    # function_call items — avoiding undocumented pairing-constraint errors.
    #
    # Graceful degradation: if the upstream rejects previous_response_id
    # (common with OpenRouter and custom proxies), we fall back to sending
    # the full history and remember not to try previous_response_id again.
    def send_responses_request(messages, model, tools, max_tokens, caching_enabled)
      use_prev_id = @last_responses_id && @responses_supports_prev_id != false
      new_messages = use_prev_id ? responses_delta_messages(messages) : messages

      body = MessageFormat::Responses.build_request_body(
        new_messages, model, tools, max_tokens, caching_enabled,
        vision_supported: @vision_supported,
        previous_response_id: use_prev_id ? @last_responses_id : nil
      )
      response = responses_connection.post("responses") { |r| r.body = body.to_json }

      # Handle errors that warrant a different request strategy
      if response.status == 400
        error_body = begin
          JSON.parse(response.body)
        rescue
          {}
        end
        error_msg = error_body.is_a?(Hash) ? error_body.dig("error", "message").to_s : ""

        # Provider doesn't support previous_response_id — retry with full history
        if error_msg.match?(/previous_response_id/i)
          @responses_supports_prev_id = false
          @last_responses_id = nil
          @last_responses_message_count = 0
          return send_responses_request(messages, model, tools, max_tokens, caching_enabled)
        end

        # Provider requires streaming
        if error_msg.match?(/stream/i)
          return send_responses_stream_request(messages, model, tools, max_tokens, caching_enabled)
        end
      end

      raise_error(response) unless response.status == 200

      # If non-streaming returned HTML (streaming-only server),
      # fallback to streaming instead of getting stuck in a retry loop.
      if html_response?(response)
        return send_responses_stream_request(messages, model, tools, max_tokens, caching_enabled)
      end

      parsed_body = safe_json_parse(response.body, context: "LLM response")

      # Track response ID and message count for next turn's delta.
      # Only save previous_response_id when the upstream confirmed it works.
      update_responses_state!(parsed_body, messages.size)

      MessageFormat::Responses.parse_response(parsed_body)
    end

    # Streaming variant of send_responses_request.  Reads the response body
    # via Faraday's on_data callback, accumulates SSE chunks, and parses
    # them into canonical format.
    #
    # Graceful degradation: same previous_response_id fallback as the
    # non-streaming path.
    def send_responses_stream_request(messages, model, tools, max_tokens, caching_enabled)
      use_prev_id = @last_responses_id && @responses_supports_prev_id != false
      new_messages = use_prev_id ? responses_delta_messages(messages) : messages

      body = MessageFormat::Responses.build_stream_request_body(
        new_messages, model, tools, max_tokens, caching_enabled,
        vision_supported: @vision_supported,
        previous_response_id: use_prev_id ? @last_responses_id : nil
      )

      chunks = []
      response = responses_connection.post("responses") do |req|
        req.body = body.to_json
        req.options.on_data = proc do |chunk, _bytes, _env|
          chunks << chunk if chunk
        end
      end

      # Handle previous_response_id rejection in streaming path too
      if response.status == 400
        error_body = begin
          reconstructed = chunks.join
          JSON.parse(reconstructed.empty? ? "{}" : reconstructed)
        rescue
          {}
        end
        error_msg = error_body.is_a?(Hash) ? error_body.dig("error", "message").to_s : ""

        if error_msg.match?(/previous_response_id/i)
          @responses_supports_prev_id = false
          @last_responses_id = nil
          @last_responses_message_count = 0
          return send_responses_stream_request(messages, model, tools, max_tokens, caching_enabled)
        end
      end

      raise_error(response, chunks: chunks) unless response.status == 200
      check_html_response(response, chunks: chunks)

      result = MessageFormat::Responses.parse_stream_response(chunks)

      # Track response ID from the final completed event for next turn's delta.
      # The response ID is embedded in the response.completed SSE event; we
      # extract it from the parsed usage data if present, or from the raw body.
      resp_id = extract_responses_id_from_stream(chunks)
      update_responses_state!(nil, messages.size, response_id: resp_id) if resp_id

      result
    end

    # Parse a simple (non-tool) Responses API response into text.
    def parse_simple_responses_response(response)
      raise_error(response) unless response.status == 200
      parsed_body = safe_json_parse(response.body, context: "LLM response")
      output = parsed_body["output"] || []
      output.select { |item| item["type"] == "message" }
            .flat_map { |item| item["content"] || [] }
            .select { |c| c["type"] == "output_text" }
            .map { |c| c["text"] }
            .join
    end

    # Streaming variant of parse_simple_responses_response.  Returns accumulated text.
    def parse_simple_responses_stream_response(model, input_items, max_tokens)
      body = { model: model, max_output_tokens: max_tokens, store: false,
               input: input_items, stream: true }

      chunks = []
      response = responses_connection.post("responses") do |req|
        req.body = body.to_json
        req.options.on_data = proc do |chunk, _bytes, _env|
          chunks << chunk if chunk
        end
      end

      raise_error(response, chunks: chunks) unless response.status == 200
      check_html_response(response, chunks: chunks)

      parsed = MessageFormat::Responses.parse_stream_response(chunks)
      parsed[:content] || ""
    end

    # ── Prompt caching helpers ────────────────────────────────────────────────

    # Add cache_control markers to the last 2 messages in the array.
    #
    # Why 2 markers:
    #   Turn N   — marks messages[-2] and messages[-1]; server caches prefix up to [-1]
    #   Turn N+1 — messages[-2] is Turn N's last message (still marked) → cache READ hit;
    #              messages[-1] is the new message (marked) → cache WRITE for Turn N+2
    #
    # With only 1 marker (old behavior): Turn N marks messages[-1]; in Turn N+1 that same
    # message is now [-2] and carries no marker → server sees a different prefix → cache MISS.
    #
    # Compression instructions (system_injected: true) are skipped — we never want to cache
    # those ephemeral injection messages.
    def apply_message_caching(messages)
      return messages if messages.empty?

      # Collect up to 2 candidate indices from the tail, skipping compression instructions.
      candidate_indices = []
      (messages.length - 1).downto(0) do |i|
        break if candidate_indices.length >= 2

        candidate_indices << i unless is_compression_instruction?(messages[i])
      end

      messages.map.with_index do |msg, idx|
        candidate_indices.include?(idx) ? add_cache_control_to_message(msg) : msg
      end
    end

    # Wrap or extend the message's content with a cache_control marker.
    def add_cache_control_to_message(msg)
      content = msg[:content]

      content_array = case content
                      when String
                        [{ type: "text", text: content, cache_control: { type: "ephemeral" } }]
                      when Array
                        content.map.with_index do |block, idx|
                          idx == content.length - 1 ? block.merge(cache_control: { type: "ephemeral" }) : block
                        end
                      else
                        return msg
                      end

      msg.merge(content: content_array)
    end

    def is_compression_instruction?(message)
      message.is_a?(Hash) && message[:system_injected] == true
    end

    # ── HTTP connections ──────────────────────────────────────────────────────

    # Bedrock Converse API endpoint path for a given model ID.
    def bedrock_endpoint(model)
      "/model/#{model}/converse"
    end

    def bedrock_connection
      @bedrock_connection ||= Faraday.new(url: @base_url) do |conn|
        conn.headers["Content-Type"]  = "application/json"
        conn.headers["Authorization"] = "Bearer #{@api_key}"
        conn.options.timeout      = 300
        conn.options.open_timeout = 10
        conn.ssl.verify           = false
        conn.adapter Faraday.default_adapter
      end
    end

    def openai_connection
      @openai_connection ||= Faraday.new(url: @base_url) do |conn|
        conn.headers["Content-Type"]  = "application/json"
        conn.headers["Authorization"] = "Bearer #{@api_key}"
        conn.options.timeout      = 300
        conn.options.open_timeout = 10
        conn.ssl.verify           = false
        conn.adapter Faraday.default_adapter
      end
    end

    # Responses API connection shares the same auth and headers as Chat
    # Completions (Bearer token, /v1 base) but POSTs to /v1/responses
    # instead of /v1/chat/completions.
    def responses_connection
      @responses_connection ||= Faraday.new(url: @base_url) do |conn|
        conn.headers["Content-Type"]  = "application/json"
        conn.headers["Authorization"] = "Bearer #{@api_key}"
        conn.options.timeout      = 300
        conn.options.open_timeout = 10
        conn.ssl.verify           = false
        conn.adapter Faraday.default_adapter
      end
    end

    # ── Responses API state tracking ───────────────────────────────────────────

    # Return only the messages that are new since the last Responses API call.
    # When @last_responses_id is set, the API already knows about all messages
    # up to @last_responses_message_count (via previous_response_id).  Only
    # send the tail — typically the latest tool results and any new user messages.
    private def responses_delta_messages(messages)
      return messages unless @last_responses_id

      messages[@last_responses_message_count..] || []
    end

    # Update the Responses API tracking state after a successful response.
    # Called from both the non-streaming (has parsed_body with "id") and
    # streaming (has explicit response_id) paths.
    private def update_responses_state!(parsed_body, message_count, response_id: nil)
      @last_responses_message_count = message_count
      @last_responses_id = response_id || parsed_body&.dig("id")
    end

    # Extract the response ID from a streaming Responses API response.
    # The ID is in the response.completed SSE event's data.response.id field.
    private def extract_responses_id_from_stream(chunks)
      body = chunks.join
      # Look for "id":"resp_xxx" in the response.completed event data
      body.each_line do |line|
        next unless line.start_with?("data: ")

        data_str = line[6..]
        next if data_str.strip == "[DONE]"

        parsed = begin
          JSON.parse(data_str)
        rescue JSON::ParserError
          nil
        end
        next unless parsed.is_a?(Hash)

        resp = parsed["response"]
        return resp["id"] if resp.is_a?(Hash) && resp["id"]
      end
      nil
    end

    # Reset Responses API state — called when switching models or on error
    # that requires a fresh conversation start.
    def reset_responses_state!
      @last_responses_id = nil
      @last_responses_message_count = 0
    end

    def anthropic_connection
      @anthropic_connection ||= Faraday.new(url: @base_url) do |conn|
        conn.headers["Content-Type"]   = "application/json"
        conn.headers["x-api-key"]      = @api_key
        conn.headers["anthropic-version"] = "2023-06-01"
        conn.headers["anthropic-dangerous-direct-browser-access"] = "true"
        # OpenRouter's /v1/messages endpoint authenticates with a Bearer
        # token (the OpenRouter API key), not Anthropic's x-api-key. We send
        # both so the same connection code works for direct Anthropic and
        # for OpenRouter-proxied Claude — each endpoint ignores the header
        # it doesn't recognise.
        if @provider_id == "openrouter"
          conn.headers["Authorization"] = "Bearer #{@api_key}"
        end
        # Moonshot's Kimi Code (Coding Plan) endpoint enforces a User-Agent
        # prefix whitelist limited to first-party coding agents (Kimi CLI,
        # Claude Code, Roo Code, Kilo Code, ...). Requests with the default
        # Faraday UA are rejected with HTTP 403 access_terminated_error,
        # despite a valid API key. We send a Claude Code-shaped UA here
        # because openclacky talks to this endpoint over the same Anthropic
        # /v1/messages protocol that Claude Code uses, so the UA matches the
        # wire-level behaviour. Hardcoding rather than exposing as a config
        # field is intentional: the only UAs known to pass the gate are the
        # whitelisted-client formats, and the project's preset registry is
        # the single source of truth for provider-specific quirks (mirroring
        # how the openrouter Bearer-fallback above is hardcoded).
        if @provider_id == "kimi-coding"
          conn.headers["User-Agent"] = "claude-cli/1.0.51 (external, cli)"
        end
        conn.options.timeout      = 300
        conn.options.open_timeout = 10
        conn.ssl.verify           = false
        conn.adapter Faraday.default_adapter
      end
    end

    # Correct relative path for the Anthropic /v1/messages endpoint, accounting
    # for whether the configured base_url already includes a "/v1" segment.
    #
    # Examples:
    #   base_url = "https://api.anthropic.com"         → "v1/messages"
    #   base_url = "https://openrouter.ai/api/v1"      → "messages"
    #   base_url = "https://openrouter.ai/api/v1/"     → "messages"
    #
    # Without this, OpenRouter would receive POST /api/v1/v1/messages → 404
    # (HTML error page), which bubbles up as the infamous
    # "Invalid API endpoint or server error (received HTML instead of JSON)".
    private def anthropic_messages_path
      base = @base_url.to_s.chomp("/")
      base.end_with?("/v1") ? "messages" : "v1/messages"
    end

    # ── Error handling ────────────────────────────────────────────────────────

    def handle_test_response(response)
      return { success: true } if response.status == 200

      error_body = JSON.parse(response.body) rescue nil
      { success: false, error: extract_error_message(error_body, response.body) }
    end

    # When streaming (on_data callback consumed response.body), chunks contain
    # the raw response data that would otherwise be in response.body.
    def raise_error(response, chunks: nil)
      raw_body = response.body.to_s
      # Streaming: response.body may be empty (data consumed by on_data callback).
      # Reconstruct the body from chunks to extract a meaningful error message.
      if (raw_body.nil? || raw_body.strip.empty?) && chunks && chunks.any?
        raw_body = chunks.join
      end
      error_body    = JSON.parse(raw_body) rescue nil
      error_message = extract_error_message(error_body, raw_body)

      case response.status
      when 400
        # Well-behaved APIs (Anthropic, OpenAI) never put quota/availability issues in 400.
        # However, some proxy/relay providers do — so we inspect the message first.
        # Also, Bedrock returns ThrottlingException as 400 instead of 429.
        if error_message.match?(/ThrottlingException|unavailable|quota/i)
          hint = error_message.match?(/quota/i) ? " (possibly out of credits)" : ""
          raise RetryableError, "[LLM] Rate limit or service issue: #{error_message}#{hint}"
        end

        # True bad request — our message was malformed. Roll back history so the
        # broken message is not replayed on the next user turn.
        raise BadRequestError, "[LLM] Client request error: #{error_message}"
      when 401 then raise AgentError, "[LLM] Invalid API key"
      when 402 then raise AgentError, "[LLM] Billing or payment issue (possibly out of credits): #{error_message}"
      when 403 then raise AgentError, "[LLM] Access denied: #{error_message}"
      when 404 then raise AgentError, "[LLM] API endpoint not found: #{error_message}"
      when 429 then raise RetryableError, "[LLM] Rate limit exceeded, please wait a moment"
      when 500..599 then raise RetryableError, "[LLM] Service temporarily unavailable (#{response.status}), retrying..."
      else raise AgentError, "[LLM] Unexpected error (#{response.status}): #{error_message}"
      end
    end

    # Raise a friendly error if the response body is HTML (e.g. gateway error page returned with 200).
    # When streaming, the response body is consumed by the on_data callback and ends up empty —
    # in that case we also check the accumulated chunks for HTML content.
    def check_html_response(response, chunks: nil)
      body = response.body.to_s.lstrip
      if html_data?(body)
        raise RetryableError, "[LLM] Service temporarily unavailable (received HTML error page), retrying..."
      end
      # Streaming: response.body may be empty (data consumed by on_data callback).
      # Check the first accumulated chunk to catch HTML that was delivered via the stream.
      if chunks
        first_chunk = chunks.first.to_s.lstrip
        if html_data?(first_chunk)
          raise RetryableError, "[LLM] Service temporarily unavailable (received HTML error page via stream), retrying..."
        end
      end
    end

    # Returns true if a string looks like HTML (starts with DOCTYPE or html tag).
    private def html_data?(str)
      str.start_with?("<!DOCTYPE", "<!doctype", "<html", "<HTML")
    end

    # Returns true if the response body looks like HTML (not JSON).
    # Used to detect when the Chat Completions endpoint doesn't exist on
    # a server — in that case we can fall back to Responses API.
    private def html_response?(response)
      body = response.body.to_s.lstrip
      html_data?(body)
    end

    def extract_error_message(error_body, raw_body)
      if raw_body.is_a?(String) && raw_body.strip.start_with?("<!DOCTYPE", "<html")
        return "Invalid API endpoint or server error (received HTML instead of JSON)"
      end

      return raw_body unless error_body.is_a?(Hash)

      error_body["upstreamMessage"]&.then { |m| return m unless m.empty? }
      error_body.dig("error", "message")&.then { |m| return m } if error_body["error"].is_a?(Hash)
      error_body["message"]&.then             { |m| return m }
      error_body["error"].is_a?(String) ? error_body["error"] : (raw_body.to_s[0..200] + (raw_body.to_s.length > 200 ? "..." : ""))
    end

    # Parse JSON with user-friendly error messages.
    # @param json_string [String] the JSON string to parse
    # @param context [String] a description of what's being parsed (e.g., "LLM response")
    # @return [Hash, Array] the parsed JSON
    # @raise [RetryableError] if parsing fails (indicates a malformed LLM response)
    def safe_json_parse(json_string, context: "response")
      JSON.parse(json_string)
    rescue JSON::ParserError => e
      # Transform technical JSON parsing errors into user-friendly messages.
      # These are usually caused by:
      #   1. Incomplete/truncated LLM response (network issue, timeout)
      #   2. LLM service returned malformed data
      #   3. Proxy/gateway corruption
      error_detail = if json_string.to_s.strip.empty?
        "received empty response"
      elsif json_string.to_s.bytesize > 500
        "response was truncated or malformed (#{json_string.to_s.bytesize} bytes received)"
      else
        "response format is invalid"
      end

      raise RetryableError, "[LLM] Failed to parse #{context}: #{error_detail}. " \
                           "This usually means the AI service returned incomplete or corrupted data. " \
                           "The request will be retried automatically."
    end

    # ── Utilities ─────────────────────────────────────────────────────────────

    def deep_clone(obj)
      case obj
      when Hash  then obj.each_with_object({}) { |(k, v), h| h[k] = deep_clone(v) }
      when Array then obj.map { |item| deep_clone(item) }
      else obj
      end
    end
  end
end
