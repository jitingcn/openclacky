# frozen_string_literal: true

# logger was removed from Ruby 4.0 stdlib; faraday needs it
require "logger"

require "faraday"
require "json"
require "digest"
require "securerandom"
require "openai"

module Clacky
  class Client
    MAX_RETRIES = 10
    RETRY_DELAY = 5 # seconds

    # Stable identity for channel affinity — ensures the aggregation proxy
    # routes requests from the same session to the same upstream, enabling
    # prompt cache hits and better performance.
    #
    # Codex CLI sends  prompt_cache_key: "ses_xxx" (per-conversation)
    # Claude Code sends metadata.user_id: hash(device_id + account_uuid + session_id)
    #
    # We generate both on first request per session and reuse them.
    attr_reader :cache_affinity_session_id, :cache_affinity_device_id, :cache_affinity_user_id

    def initialize(api_key, base_url:, model:, anthropic_format: false, api_type: nil, stream: nil)
      @api_key = api_key
      @base_url = base_url
      @model = model
      @stream = stream  # true = always stream, false = never stream, nil = auto

      # ── Cache affinity identity ──────────────────────────────────────────────
      # Generate stable identifiers for proxy channel affinity.
      # These mimic Codex CLI's prompt_cache_key and Claude Code's metadata.user_id.
      #
      # Claude Code (>= 2.1.78) sends metadata.user_id as a JSON string:
      #   {"device_id":"sha256hex...","account_uuid":"","session_id":"..."}
      # The proxy uses gjson:metadata.user_id for channel stickiness.
      @cache_affinity_session_id = SecureRandom.hex(16)
      @cache_affinity_device_id  = Digest::SHA256.hexdigest("claude_user_#{Digest::SHA256.hexdigest(api_key.to_s)[0, 16]}")
      @cache_affinity_user_id    = JSON.generate({
        device_id: @cache_affinity_device_id,
        account_uuid: "",
        session_id: @cache_affinity_session_id
      })

      # Responses API state is now managed by the OpenAI SDK internally.
      # previous_response_id and delta messages tracking are no longer needed.

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
      #
      # IMPORTANT: when api_type is explicitly set to "anthropic-messages",
      # the case statement above already set @use_anthropic_format = true.
      # Do NOT override it with provider detection.
      unless resolved_api_type == "anthropic-messages"
        provider_prefers_anthropic = provider_id &&
                                     Providers.anthropic_format_for_model?(provider_id, @model)
        @use_anthropic_format = provider_prefers_anthropic || anthropic_format
      end

      # Remember the provider id so we can tune connection headers below
      # (OpenRouter's /v1/messages accepts either Bearer or x-api-key, but
      # some OpenRouter-compatible relays only honour Bearer — send both).
      @provider_id = provider_id

      # Determine vision support once at construction time.
      # Non-vision models (DeepSeek, Kimi, MiniMax, etc.) reject image_url
      # content blocks; the conversion layer strips them when this is false.
      @vision_supported = Providers.supports?(provider_id, :vision, model_name: @model)

      # ── OpenAI SDK client for Responses API ──────────────────────────────────
      # When the resolved api_type is "openai-responses", we delegate all
      # Responses API calls to the official openai-ruby SDK instead of
      # hand-rolling HTTP requests and SSE parsing.  The SDK provides proper
      # streaming event parsing, type-safe responses, and built-in retries.
      #
      # For non-Responses paths (Chat Completions, Anthropic, Bedrock) we
      # continue using Faraday directly — the SDK is only wired up here.
      @openai_sdk_client = nil
      if @use_responses
        @openai_sdk_client = build_openai_client
      end
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
        begin
          # Use stream_raw to test connection — works with both streaming-only
          # and non-streaming proxies. Just consume the first event.
          stream = @openai_sdk_client.responses.stream_raw(
            model: model,
            max_output_tokens: 16,
            store: false,
            input: [{ role: "user", content: "hi" }]
          )
          # Consume at least one event to confirm the connection works
          stream.each { |_event| break }
          return { success: true }
        rescue OpenAI::Errors::APIStatusError => e
          return { success: false, error: e.message }
        rescue => e
          return { success: false, error: e.message }
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
        # Default to streaming first unless user explicitly disables it.
        # When stream is forced on (true), streaming failures propagate.
        # When stream is nil (auto), fall back to non-streaming gracefully.
        unless @stream == false
          begin
            return parse_simple_anthropic_stream_response(model, messages, max_tokens)
          rescue StandardError
            raise if @stream == true
            # @stream == nil: streaming failed, fall through to non-streaming below
          end
        end

        # Non-streaming path — either @stream == false or streaming fallback
        body     = MessageFormat::Anthropic.build_request_body(messages, model, [], max_tokens, false)
        inject_cache_affinity!(body, :anthropic)
        response = anthropic_connection.post(anthropic_messages_path) { |r| r.body = body.to_json }
        # Fall back to Responses API (via OpenAI SDK) when Anthropic endpoint
        # requires streaming or returns HTML (endpoint doesn't exist).
        if response.status == 400 || html_response?(response)
          error_body = begin; JSON.parse(response.body); rescue; {}; end
          error_msg = error_body.is_a?(Hash) ? error_body.dig("error", "message").to_s : ""
          if error_msg.match?(/stream/i) || html_response?(response)
            @use_responses = true
            @openai_sdk_client ||= build_openai_client
            begin
              sdk_resp = @openai_sdk_client.responses.create(
                model: model, max_output_tokens: max_tokens,
                store: false, input: messages,
                prompt_cache_key: @cache_affinity_session_id
              )
              return extract_response_text(sdk_resp)
            rescue OpenAI::Errors::BadRequestError => e
              if e.message.include?("stream") || e.message.include?("Stream")
                sdk_resp = stream_responses_via_sdk(
                  model: model, max_output_tokens: max_tokens,
                  store: false, input: messages,
                  prompt_cache_key: @cache_affinity_session_id
                )
                return extract_response_text(sdk_resp)
              end
              raise map_sdk_error(e)
            rescue OpenAI::Errors::APIStatusError => e
              raise map_sdk_error(e)
            end
          end
        end
        parse_simple_anthropic_response(response)
      elsif @use_responses
        # Use OpenAI SDK for Responses API.
        begin
          sdk_resp = @openai_sdk_client.responses.create(
            model: model,
            max_output_tokens: max_tokens,
            store: false,
            input: messages,
            prompt_cache_key: @cache_affinity_session_id
          )
          extract_response_text(sdk_resp)
        rescue OpenAI::Errors::BadRequestError => e
          # Proxy requires streaming — fallback to stream + get_final_response
          if e.message.include?("stream") || e.message.include?("Stream")
            sdk_resp = stream_responses_via_sdk(
              model: model, max_output_tokens: max_tokens,
              store: false, input: messages,
              prompt_cache_key: @cache_affinity_session_id
            )
            extract_response_text(sdk_resp)
          else
            raise map_sdk_error(e)
          end
        rescue OpenAI::Errors::APIStatusError => e
          raise map_sdk_error(e)
        end
      else
        # Default to streaming first unless user explicitly disables it.
        # When stream is forced on (true), streaming failures propagate.
        # When stream is nil (auto), fall back to non-streaming gracefully.
        unless @stream == false
          begin
            return parse_simple_openai_stream_response(model, messages, max_tokens)
          rescue StandardError
            raise if @stream == true
            # @stream == nil: streaming failed, fall through to non-streaming below
          end
        end

        # Non-streaming path — either @stream == false or streaming fallback
        body = { model: model, max_tokens: max_tokens, messages: messages }
        inject_cache_affinity!(body, :openai)

        # Try non-streaming with a short timeout (15s).  Some servers
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
    def send_messages_with_tools(messages, model:, tools:, max_tokens:, enable_caching: false, prompt_caching: nil)
      caching_enabled = enable_caching && supports_prompt_caching?(model, prompt_caching)
      cloned = deep_clone(messages)

      t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      response =
        if @use_responses
          send_sdk_responses_request(cloned, model, tools, max_tokens, caching_enabled)
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
    def supports_prompt_caching?(_model, explicit_flag = nil)
      # 1) Per-model explicit config wins (true/false in config.yml)
      return explicit_flag unless explicit_flag.nil?

      # 2) Fallback: Anthropic Messages API supports cache_control as a standard
      #    protocol feature; OpenAI Chat Completions only supports it for
      #    specific backends (OpenRouter → Anthropic passthrough).
      @use_anthropic_format
    end


    # ── Cache affinity injection ───────────────────────────────────────────────
    #
    # Inject session identity fields into the request body so that aggregation
    # proxies (e.g. new-api, one-api) can route requests from the same session
    # to the same upstream channel.  Without these fields, the proxy randomly
    # assigns upstreams, breaking prompt cache affinity.
    #
    # Codex CLI behavior (Responses API / Chat Completions):
    #   - Body: { prompt_cache_key: "ses_xxx", ... }
    #   - The proxy matches on gjson:prompt_cache_key to group requests
    #
    # Claude Code behavior (Anthropic Messages API):
    #   - Body: { metadata: { user_id: "sha256hash..." }, ... }
    #   - The proxy matches on gjson:metadata.user_id to group requests
    #
    # @param body [Hash] the request body hash (mutated in place)
    # @param api_format [Symbol] :openai, :anthropic, :responses, or :bedrock
    private def inject_cache_affinity!(body, api_format)
      case api_format
      when :responses, :openai
        # Codex CLI sends prompt_cache_key as a stable session identifier.
        # Aggregation proxies use this field (via gjson path) for channel
        # affinity — all requests with the same key hit the same upstream.
        body[:prompt_cache_key] = @cache_affinity_session_id
      when :anthropic
        # Claude Code (>= 2.1.78) sends metadata.user_id as a JSON-encoded
        # string: {"device_id":"sha256...","account_uuid":"","session_id":"..."}
        # Proxies match on gjson:metadata.user_id for stickiness.
        body[:metadata] ||= {}
        body[:metadata][:user_id] = @cache_affinity_user_id
      when :bedrock
        # Bedrock uses AWS session management — no injection needed
      end
      body
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

      # Default to streaming first unless user explicitly disables it.
      # When stream is forced on (true), streaming failures propagate.
      # When stream is nil (auto), fall back to non-streaming gracefully.
      unless @stream == false
        begin
          return send_anthropic_stream_request(messages, model, tools, max_tokens, caching_enabled)
        rescue StandardError
          raise if @stream == true
          # @stream == nil: streaming failed, fall through to non-streaming below
        end
      end

      # Non-streaming path — either @stream == false or streaming fallback
      body     = MessageFormat::Anthropic.build_request_body(messages, model, tools, max_tokens, caching_enabled)
      inject_cache_affinity!(body, :anthropic)
      response = anthropic_connection.post(anthropic_messages_path) { |r|
        r.body = body.to_json
        r.headers["x-client-request-id"] = SecureRandom.uuid
      }

      # Some providers require streaming even for Anthropic-format endpoints.
      # When we get a 400 with "stream" in the error, fall back to Responses API via SDK.
      if response.status == 400
        error_body = begin
          JSON.parse(response.body)
        rescue
          {}
        end
        error_msg = error_body.is_a?(Hash) ? error_body.dig("error", "message").to_s : ""
        if error_msg.match?(/stream/i)
          @use_responses = true
          @openai_sdk_client ||= build_openai_client
          return send_sdk_responses_request(messages, model, tools, max_tokens, caching_enabled)
        end
      end

      # When Anthropic endpoint returns HTML, it doesn't exist — fall back to Responses API via SDK
      if html_response?(response)
        @use_responses = true
        @openai_sdk_client ||= build_openai_client
        return send_sdk_responses_request(messages, model, tools, max_tokens, caching_enabled)
      end

      raise_error(response) unless response.status == 200
      check_html_response(response)
      parsed_body = safe_json_parse(response.body, context: "LLM response")
      MessageFormat::Anthropic.parse_response(parsed_body)
    end

    # Streaming variant of send_anthropic_request.  Reads the response body
    # via Faraday's on_data callback, accumulates Anthropic SSE chunks, and
    # parses them into canonical format.
    def send_anthropic_stream_request(messages, model, tools, max_tokens, caching_enabled)
      # Apply cache_control to the message that marks the cache breakpoint
      messages = apply_message_caching(messages) if caching_enabled

      body = MessageFormat::Anthropic.build_stream_request_body(messages, model, tools, max_tokens, caching_enabled)
      inject_cache_affinity!(body, :anthropic)

      chunks = []
      response = anthropic_connection.post(anthropic_messages_path) do |req|
        req.body = body.to_json
        req.headers["x-client-request-id"] = SecureRandom.uuid
        req.options.on_data = proc do |chunk, _bytes, _env|
          chunks << chunk if chunk
        end
      end

      # Handle errors: some providers require streaming via Responses API
      if response.status == 400
        error_body = begin
          reconstructed = chunks.join
          JSON.parse(reconstructed.empty? ? "{}" : reconstructed)
        rescue
          {}
        end
        error_msg = error_body.is_a?(Hash) ? error_body.dig("error", "message").to_s : ""
        if error_msg.match?(/stream/i)
          @use_responses = true
          @openai_sdk_client ||= build_openai_client
          return send_sdk_responses_request(messages, model, tools, max_tokens, caching_enabled)
        end
      end

      raise_error(response, chunks: chunks) unless response.status == 200

      # HTML response via stream means the endpoint doesn't support streaming
      # (e.g. proxy returned an error page). Don't raise RetryableError here —
      # let the caller's streaming-first fallback degrade to non-streaming
      # instead of looping forever in call_llm's retry handler.
      first_chunk = chunks.first.to_s.lstrip
      if html_data?(first_chunk)
        raise StreamFallbackError, "Anthropic streaming returned HTML — falling back to non-streaming"
      end
      if html_data?(response.body.to_s.lstrip)
        raise StreamFallbackError, "Anthropic streaming returned HTML — falling back to non-streaming"
      end

      MessageFormat::Anthropic.parse_stream_response(chunks)
    end

    def parse_simple_anthropic_response(response)
      raise_error(response) unless response.status == 200
      data = safe_json_parse(response.body, context: "LLM response")
      (data["content"] || []).select { |b| b["type"] == "text" }.map { |b| b["text"] }.join("")
    end

    # Streaming variant of parse_simple_anthropic_response.  Returns accumulated text.
    def parse_simple_anthropic_stream_response(model, messages, max_tokens)
      body = MessageFormat::Anthropic.build_stream_request_body(
        messages, model, [], max_tokens, false
      )
      inject_cache_affinity!(body, :anthropic)

      chunks = []
      response = anthropic_connection.post(anthropic_messages_path) do |req|
        req.body = body.to_json
        req.options.on_data = proc do |chunk, _bytes, _env|
          chunks << chunk if chunk
        end
      end

      raise_error(response, chunks: chunks) unless response.status == 200

      # HTML via stream means endpoint doesn't support streaming — signal
      # caller to degrade to non-streaming instead of triggering RetryableError.
      first_chunk = chunks.first.to_s.lstrip
      if html_data?(first_chunk) || html_data?(response.body.to_s.lstrip)
        raise StreamFallbackError, "Anthropic streaming returned HTML — falling back to non-streaming"
      end

      parsed = MessageFormat::Anthropic.parse_stream_response(chunks)
      parsed[:content] || ""
    end

    # ── OpenAI request / response ─────────────────────────────────────────────

    def send_openai_request(messages, model, tools, max_tokens, caching_enabled)
      # Apply cache_control markers to messages when caching is enabled.
      # OpenRouter proxies Claude with the same cache_control field convention as Anthropic direct.
      messages = apply_message_caching(messages) if caching_enabled

      # Default to streaming first unless user explicitly disables it.
      # When stream is forced on (true), streaming failures propagate.
      # When stream is nil (auto), we fall back to non-streaming gracefully.
      unless @stream == false
        begin
          return send_openai_stream_request(messages, model, tools, max_tokens, caching_enabled)
        rescue StandardError
          raise if @stream == true
          # @stream == nil: streaming failed, fall through to non-streaming below
        end
      end

      body     = MessageFormat::OpenAI.build_request_body(
        messages, model, tools, max_tokens, caching_enabled,
        vision_supported: @vision_supported
      )
      inject_cache_affinity!(body, :openai)

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
      # Apply cache_control markers to messages when caching is enabled.
      # OpenRouter proxies Claude with the same cache_control field convention as Anthropic direct.
      messages = apply_message_caching(messages) if caching_enabled

      body = MessageFormat::OpenAI.build_stream_request_body(
        messages, model, tools, max_tokens, caching_enabled,
        vision_supported: @vision_supported
      )
      inject_cache_affinity!(body, :openai)

      chunks = []
      response = openai_connection.post("chat/completions") do |req|
        req.body = body.to_json
        req.options.on_data = proc do |chunk, _bytes, _env|
          chunks << chunk if chunk
        end
      end

      raise_error(response, chunks: chunks) unless response.status == 200

      # HTML via stream — degrade to non-streaming (OpenAI stream request path)
      if html_data?(chunks.first.to_s.lstrip) || html_data?(response.body.to_s.lstrip)
        raise StreamFallbackError, "OpenAI streaming returned HTML — falling back to non-streaming"
      end

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
      inject_cache_affinity!(body, :openai)

      chunks = []
      response = openai_connection.post("chat/completions") do |req|
        req.body = body.to_json
        req.options.on_data = proc do |chunk, _bytes, _env|
          chunks << chunk if chunk
        end
      end

      raise_error(response, chunks: chunks) unless response.status == 200

      # HTML via stream — degrade to non-streaming (simple OpenAI stream path)
      if html_data?(chunks.first.to_s.lstrip) || html_data?(response.body.to_s.lstrip)
        raise StreamFallbackError, "OpenAI streaming returned HTML — falling back to non-streaming"
      end

      parsed = MessageFormat::OpenAI.parse_stream_response(chunks)
      parsed[:content] || ""
    end

    # ── Responses API via OpenAI SDK ──────────────────────────────────────────
    #
    # All Responses API calls are delegated to the official openai-ruby SDK,
    # which handles SSE parsing, event typing, and retries correctly.
    # We only need to:
    #   1. Build the request params (reusing MessageFormat::Responses)
    #   2. Call the SDK
    #   3. Convert the SDK's typed response to our canonical hash format

    # Send a Responses API request via the OpenAI SDK.
    # Returns canonical response hash: { content:, tool_calls:, finish_reason:, usage:, raw_api_usage: }
    #
    # Strategy: try non-streaming first (SDK's create).  Some proxies
    # (e.g. 211server) require stream: true and reject non-streaming with
    # 400.  When that happens, fall back to SDK's stream() + until_done.
    def send_sdk_responses_request(messages, model, tools, max_tokens, caching_enabled)
      input_items = MessageFormat::Responses.build_input_items(
        messages, vision_supported: @vision_supported
      )

      params = {
        model: model,
        input: input_items,
        max_output_tokens: max_tokens,
        store: false,
        prompt_cache_key: @cache_affinity_session_id
      }

      if tools&.any?
        params[:tools] = tools.map { |t| MessageFormat::Responses.convert_tool_to_responses_format(t) }
        params[:tool_choice] = "auto"
      end

      begin
        # When stream: true is configured (e.g. streaming-only proxies),
        # skip the non-streaming create() attempt entirely.
        if @stream
          result = stream_responses_via_sdk(params)
          result.is_a?(Hash) ? result : convert_sdk_response(result)
        else
          sdk_resp = @openai_sdk_client.responses.create(params)
          convert_sdk_response(sdk_resp)
        end
      rescue OpenAI::Errors::BadRequestError => e
        # Proxy requires streaming — fall back to SDK's stream_raw
        if e.message.include?("stream") || e.message.include?("Stream")
          result = stream_responses_via_sdk(params)
          result.is_a?(Hash) ? result : convert_sdk_response(result)
        else
          raise map_sdk_error(e)
        end
      rescue OpenAI::Errors::APIStatusError => e
        raise map_sdk_error(e)
      end
    end

    # Fallback for streaming-only proxies: use SDK's stream() to collect the
    # full response, then extract the final Response object.
    #
    # Some proxies send `response.complete` instead of `response.completed`,
    # which means SDK's ResponseStreamState never finalizes and get_final_response
    # returns an empty output.  In that case, we manually accumulate the text
    # and tool calls from the raw stream events.
    private def stream_responses_via_sdk(params)
      # Always use stream_raw directly — don't waste an API call on stream()
      # + get_final_response first.  Some proxies send `response.complete`
      # instead of `response.completed`, so SDK's ResponseStreamState never
      # finalizes, making get_final_response always fail.  Using stream_raw
      # avoids the extra (wasted) HTTP request.
      content_parts = []
      tool_calls    = []
      usage_data    = nil
      resp_id       = nil

      raw_stream = @openai_sdk_client.responses.stream_raw(params)
      raw_stream.each do |event|
        case event.type
        when :"response.output_text.delta"
          content_parts << event.delta if event.delta
        when :"response.output_text.done"
          # Full text — authoritative, use this over accumulated deltas
          content_parts.clear
          content_parts << event.text if event.text
        when :"response.output_item.done"
          # The completed output item has all fields including call_id.
          # Use this instead of function_call_arguments.done which lacks call_id.
          item = event.item
          if item.is_a?(OpenAI::Models::Responses::ResponseFunctionToolCall)
            tool_calls << {
              id:        item.call_id || item.id,
              type:      "function",
              name:      item.name,
              arguments: item.arguments.to_s
            }
          end
        when :"response.completed", :"response.complete"
          # response.complete is a non-standard proxy variant
          if event.respond_to?(:response) && event.response
            resp_id = event.response.id if event.response.respond_to?(:id)
            if event.response.respond_to?(:usage) && event.response.usage
              usage_data = event.response.usage
            end
          end
        end
      end

      build_canonical_from_stream(content_parts, tool_calls, usage_data, resp_id)
    end

    # Build the canonical response hash directly from accumulated stream data,
    # bypassing convert_sdk_response (which requires SDK typed objects).
    private def build_canonical_from_stream(content_parts, tool_calls, usage_data, resp_id)
      content    = content_parts.empty? ? nil : content_parts.join
      tool_calls = nil if tool_calls.empty?
      finish_reason = tool_calls && !tool_calls.empty? ? "tool_calls" : "stop"

      # Extract token counts — same logic as convert_sdk_response
      raw_usage_h = usage_data&.deep_to_h || {}

      prompt_tokens = usage_data&.input_tokens ||
                      raw_usage_h[:prompt_tokens] ||
                      raw_usage_h["prompt_tokens"].to_i
      completion_tokens = usage_data&.output_tokens ||
                          raw_usage_h[:completion_tokens] ||
                          raw_usage_h["completion_tokens"].to_i
      total_tokens = usage_data&.total_tokens ||
                     raw_usage_h[:total_tokens] ||
                     prompt_tokens.to_i + completion_tokens.to_i

      cached_tokens = usage_data&.input_tokens_details&.cached_tokens ||
                      raw_usage_h.dig(:input_tokens_details, :cached_tokens) ||
                      raw_usage_h.dig(:prompt_tokens_details, :cached_tokens).to_i

      usage = {
        prompt_tokens:     prompt_tokens.to_i,
        completion_tokens: completion_tokens.to_i,
        total_tokens:      total_tokens.to_i
      }
      usage[:cache_read_input_tokens] = cached_tokens.to_i if cached_tokens.to_i > 0

      raw_usage = {
        input_tokens:  prompt_tokens.to_i,
        output_tokens: completion_tokens.to_i,
        total_tokens:  total_tokens.to_i
      }
      raw_usage.merge!(raw_usage_h)

      {
        id:               resp_id,
        content:          content,
        tool_calls:       tool_calls,
        finish_reason:    finish_reason,
        usage:            usage,
        raw_api_usage:    raw_usage,
        model:            nil,
        response_object:  nil
      }
    end

    # Convert an OpenAI::Responses::Response object to our canonical hash.
    #
    # SDK response shape:
    #   response.output → [Message items, FunctionToolCall items, ...]
    #   response.usage  → { input_tokens:, output_tokens:, total_tokens: }
    #   response.id     → "resp_xxx"
    private def convert_sdk_response(sdk_resp)
      content_parts = []
      tool_calls    = []
      reasoning     = nil

      sdk_resp.output.each do |item|
        case item
        when OpenAI::Models::Responses::ResponseOutputMessage
          next unless item.role == :assistant

          item.content.each do |part|
            case part
            when OpenAI::Models::Responses::ResponseOutputText
              content_parts << part.text
            end
          end
        when OpenAI::Models::Responses::ResponseFunctionToolCall
          tool_calls << {
            id:        item.call_id || item.id,
            type:      "function",
            name:      item.name,
            arguments: item.arguments.to_s
          }
        when OpenAI::Models::Responses::ResponseReasoningItem
          # Extract reasoning text if present
          if item.respond_to?(:summary) && item.summary
            reasoning = item.summary.map { |s| s.text if s.respond_to?(:text) }.compact.join("\n")
          end
        end
      end

      content    = content_parts.empty? ? nil : content_parts.join
      tool_calls = nil if tool_calls.empty?

      finish_reason = tool_calls && !tool_calls.empty? ? "tool_calls" : "stop"

      usage_data = sdk_resp.usage

      # Extract token counts from SDK's typed usage object.
      # The SDK's ResponseUsage uses `input_tokens` / `output_tokens` (Responses
      # API naming), but many proxies (OpenRouter, etc.) return Chat Completions
      # format with `prompt_tokens` / `completion_tokens` instead.  When the SDK
      # can't find the expected field it returns nil, so we fall back to the
      # raw hash via `deep_to_h` which preserves ALL keys from the API response.
      raw_usage_h = usage_data&.deep_to_h || {}

      prompt_tokens = usage_data&.input_tokens ||
                      raw_usage_h[:prompt_tokens] ||
                      raw_usage_h["prompt_tokens"].to_i
      completion_tokens = usage_data&.output_tokens ||
                          raw_usage_h[:completion_tokens] ||
                          raw_usage_h["completion_tokens"].to_i
      total_tokens = usage_data&.total_tokens ||
                     raw_usage_h[:total_tokens] ||
                     prompt_tokens.to_i + completion_tokens.to_i

      # Extract cache tokens from SDK's nested input_tokens_details.
      # The Responses API reports cached_tokens inside input_tokens_details,
      # which is the OpenAI equivalent of Anthropic's cache_read_input_tokens.
      # Proxies may also send prompt_tokens_details.cached_tokens.
      cached_tokens = usage_data&.input_tokens_details&.cached_tokens ||
                      raw_usage_h.dig(:input_tokens_details, :cached_tokens) ||
                      raw_usage_h.dig(:prompt_tokens_details, :cached_tokens).to_i

      usage = {
        prompt_tokens:     prompt_tokens.to_i,
        completion_tokens: completion_tokens.to_i,
        total_tokens:      total_tokens.to_i
      }
      usage[:cache_read_input_tokens] = cached_tokens.to_i if cached_tokens.to_i > 0

      # Build raw_api_usage hash for track_cost / CostTracker compatibility.
      # Uses symbol keys to match the convention from MessageFormat::OpenAI / Anthropic.
      raw_usage = {
        input_tokens:  prompt_tokens.to_i,
        output_tokens: completion_tokens.to_i,
        total_tokens:  total_tokens.to_i
      }
      if cached_tokens.to_i > 0
        raw_usage[:input_tokens_details] = { cached_tokens: cached_tokens.to_i }
        raw_usage[:cache_read_input_tokens] = cached_tokens.to_i
      end

      result = {
        content:       content,
        tool_calls:    tool_calls,
        finish_reason: finish_reason,
        usage:         usage,
        raw_api_usage: raw_usage
      }
      result[:reasoning_content] = reasoning if reasoning
      result
    end

    # Extract plain text from an SDK Response object (for simple/non-tool calls).
    # Extract text content from an SDK Response object or a canonical hash.
    private def extract_response_text(sdk_resp)
      # Canonical hash from build_canonical_from_stream
      return sdk_resp[:content].to_s if sdk_resp.is_a?(Hash)

      sdk_resp.output.filter_map do |item|
        next unless item.is_a?(OpenAI::Models::Responses::ResponseOutputMessage) && item.role == :assistant

        item.content.filter_map do |part|
          part.text if part.is_a?(OpenAI::Models::Responses::ResponseOutputText)
        end.join
      end.join
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

    # ── OpenAI SDK helpers ───────────────────────────────────────────────────

    # Build an OpenAI::Client instance configured to talk to the user's
    # chosen base_url with Kilo Code identity headers injected.
    #
    # The SDK uses net/http internally (with connection pooling via the
    # connection_pool gem), so we don't need Faraday for this path.
    private def build_openai_client
      OpenAI::Client.new(
        api_key: @api_key,
        base_url: @base_url,
        timeout: 300,
        max_retries: 0   # we handle retries ourselves in llm_caller
      )
    end

    # Map an OpenAI SDK error to our internal error classes so the retry
    # and fallback logic in llm_caller continues to work unchanged.
    #
    # SDK error hierarchy:
    #   OpenAI::Errors::APIStatusError
    #     ├── APIConnectionError         → RetryableError (transient)
    #     ├── BadRequestError (400)      → BadRequestError
    #     ├── AuthenticationError (401)  → AgentError
    #     ├── PermissionDeniedError (403)→ AgentError
    #     ├── NotFoundError (404)        → AgentError
    #     ├── ConflictError (409)        → RetryableError
    #     ├── UnprocessableEntityError (422) → BadRequestError
    #     ├── RateLimitError (429)       → RetryableError
    #     ├── InternalServerError (500)  → RetryableError
    private def map_sdk_error(error)
      klass, msg = case error
      when OpenAI::Errors::APIConnectionError
        [RetryableError, "[LLM] Connection error: #{error.message}"]
      when OpenAI::Errors::RateLimitError
        [RetryableError, "[LLM] Rate limit exceeded, please wait a moment"]
      when OpenAI::Errors::InternalServerError
        [RetryableError, "[LLM] Service temporarily unavailable (#{error.status}), retrying..."]
      when OpenAI::Errors::BadRequestError
        [Clacky::BadRequestError, "[LLM] Client request error: #{error.message}"]
      when OpenAI::Errors::AuthenticationError
        [AgentError, "[LLM] Invalid API key"]
      when OpenAI::Errors::PermissionDeniedError
        [AgentError, "[LLM] Access denied: #{error.message}"]
      when OpenAI::Errors::NotFoundError
        [AgentError, "[LLM] API endpoint not found: #{error.message}"]
      else
        if error.status && error.status >= 500
          [RetryableError, "[LLM] Service temporarily unavailable (#{error.status}), retrying..."]
        elsif error.status && error.status == 429
          [RetryableError, "[LLM] Rate limit exceeded, please wait a moment"]
        elsif error.status && error.status == 402
          [AgentError, "[LLM] Billing or payment issue: #{error.message}"]
        else
          [AgentError, "[LLM] API error (#{error.status}): #{error.message}"]
        end
      end
      klass.new(msg)
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

    # Kilo Code client identity constants for aggregation proxy channel affinity.
    # Kilo Code's DEFAULT_HEADERS (from @/kilocode/const.ts):
    #   HTTP-Referer: https://kilocode.ai
    #   X-Title: Kilo Code
    #   User-Agent: Kilo-Code/{version}
    #
    # The AI SDK appends its own suffix to the User-Agent at request time via
    # withUserAgentSuffix(), producing:
    #   Kilo-Code/7.2.42 ai-sdk/provider-utils/4.0.23 runtime/node.js/v22.11.0
    KILO_CODE_VERSION = "7.2.42"
    AI_SDK_PROVIDER_UTILS_VERSION = "4.0.23"
    AI_SDK_PROVIDER_UA = "ai-sdk/provider-utils/#{AI_SDK_PROVIDER_UTILS_VERSION}".freeze
    NODE_RUNTIME_UA = "runtime/node.js/v22.11.0"
    KILO_CODE_UA = "Kilo-Code/#{KILO_CODE_VERSION} #{AI_SDK_PROVIDER_UA} #{NODE_RUNTIME_UA}".freeze
    private_constant :AI_SDK_PROVIDER_UA, :AI_SDK_PROVIDER_UTILS_VERSION, :NODE_RUNTIME_UA

    # Shared Kilo Code client identity headers applied to OpenAI and Responses
    # connections.  These mimic what a real Kilo Code CLI sends, helping
    # aggregation proxies identify and route our requests.
    def kilo_code_headers
      {
        "HTTP-Referer" => "https://kilocode.ai",
        "X-Title"      => "Kilo Code",
        "User-Agent"   => KILO_CODE_UA
      }
    end

    def openai_connection
      @openai_connection ||= Faraday.new(url: @base_url) do |conn|
        conn.headers["Content-Type"]  = "application/json"
        conn.headers["Authorization"] = "Bearer #{@api_key}"
        kilo_code_headers.each { |k, v| conn.headers[k] = v }
        conn.options.timeout      = 300
        conn.options.open_timeout = 10
        conn.ssl.verify           = false
        conn.adapter Faraday.default_adapter
      end
    end

    # Reset Responses API state — called when switching models or on error
    # that requires a fresh conversation start.
    # NOTE: With the OpenAI SDK, state tracking is handled internally by the
    # SDK's response objects. This method is kept as a no-op for compatibility.
    def reset_responses_state!
      # no-op: SDK manages response state internally
    end

    def anthropic_connection
      @anthropic_connection ||= Faraday.new(url: @base_url) do |conn|
        conn.headers["Content-Type"]   = "application/json"
        conn.headers["Accept"]         = "application/json"
        conn.headers["x-api-key"]      = @api_key
        conn.headers["Authorization"]   = "Bearer #{@api_key}"
        conn.headers["anthropic-version"] = "2023-06-01"
        conn.headers["anthropic-dangerous-direct-browser-access"] = "true"
        # Claude Code-shaped client identity headers help certain aggregation
        # / coding channels route Anthropic-compatible traffic correctly.
        # Keep these generic headers enabled by default, while preserving
        # provider-specific auth / UA overrides below.
        conn.headers["x-app"] = "cli"
        conn.headers["X-Claude-Code-Session-Id"] = @cache_affinity_session_id
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
