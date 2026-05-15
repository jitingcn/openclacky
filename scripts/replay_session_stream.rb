#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "optparse"
require "pathname"
require "fileutils"

require_relative "../lib/clacky"

module Clacky
  class SessionStreamReplay
    DEFAULT_CAPTURE_DIR = File.join(Dir.home, ".clacky", "tmp").freeze

    def initialize(options)
      @options = options
    end

    def run
      session = load_session if @options[:session] || @options[:session_file]
      replay_messages, saved_assistant = resolve_replay_messages(session)

      if @options[:raw_file]
        chunks = [File.binread(@options[:raw_file])]
        request_info = {
          mode: "raw-file",
          raw_file: @options[:raw_file]
        }
      else
        client, model_name, max_tokens = build_client_from_session(session)
        request_info = {
          mode: "live-capture",
          session_id: session[:session_id],
          model: model_name,
          base_url: session.dig(:config, :model_base_url),
          max_tokens: max_tokens,
          request_body: redacted_request_body(client, replay_messages, model_name, max_tokens)
        }
        chunks = capture_live_stream(client, replay_messages, model_name, max_tokens)
        capture_path = persist_capture(chunks, session[:session_id])
        request_info[:capture_file] = capture_path
      end

      raw_events = extract_raw_events(chunks)
      observer_events = replay_observer_events(chunks)
      parser_result = MessageFormat::OpenAI.parse_stream_response(chunks)

      puts JSON.pretty_generate({
        session: session_summary(session, saved_assistant),
        replay_request: request_info,
        replay_messages: summarize_messages(replay_messages),
        raw_events: raw_events,
        observer_events: observer_events,
        parser_result: summarize_parser_result(parser_result)
      })
    end

    private

    def load_session
      if @options[:session_file]
        path = File.expand_path(@options[:session_file])
        raise "Session file not found: #{path}" unless File.exist?(path)

        JSON.parse(File.read(path), symbolize_names: true)
      else
        manager = SessionManager.new
        session = manager.load(@options[:session])
        raise "Session not found: #{@options[:session]}" unless session

        session
      end
    end

    def resolve_replay_messages(session)
      return [[], nil] unless session

      messages = session[:messages] || []
      assistant_index = resolve_assistant_index(messages)
      raise "No assistant message found in session" unless assistant_index

      saved_assistant = messages[assistant_index]
      [messages[0...assistant_index], saved_assistant]
    end

    def resolve_assistant_index(messages)
      explicit = @options[:assistant_index]
      return explicit if explicit && explicit >= 0 && explicit < messages.length

      (messages.length - 1).downto(0).find { |idx| messages[idx][:role] == "assistant" }
    end

    def build_client_from_session(session)
      raise "Live capture requires --session or --session-file" unless session

      session_config = session[:config] || {}
      model_name = @options[:model] || session_config[:model_name]
      base_url = @options[:base_url] || session_config[:model_base_url]
      max_tokens = (@options[:max_tokens] || session_config[:max_tokens] || 16_384).to_i
      raise "Session config missing model_name" if model_name.to_s.empty?
      raise "Session config missing model_base_url" if base_url.to_s.empty?

      config = AgentConfig.load(AgentConfig::CONFIG_FILE)
      model_entry = config.find_model_by_name_and_url(model_name, base_url)
      raise "No matching configured model for #{model_name} @ #{base_url}" unless model_entry

      client = Client.new(
        model_entry["api_key"],
        base_url: base_url,
        model: model_name,
        anthropic_format: model_entry["anthropic_format"] || false,
        anthropic_stream: model_entry.key?("anthropic_stream") ? model_entry["anthropic_stream"] : true,
        api_type: model_entry["api_type"],
        stream: true,
        thinking_enabled: model_entry.key?("thinking_enabled") ? model_entry["thinking_enabled"] : nil,
        reasoning_effort: model_entry["reasoning_effort"]
      )

      [client, model_name, max_tokens]
    end

    def redacted_request_body(client, messages, model_name, max_tokens)
      body = client.send(:build_openai_stream_request_body, messages, model_name, [], max_tokens, false)
      client.send(:inject_cache_affinity!, body, :openai)
      body
    end

    def capture_live_stream(client, messages, model_name, max_tokens)
      body = redacted_request_body(client, messages, model_name, max_tokens)
      chunks = []

      response = client.send(:openai_connection).post("chat/completions") do |req|
        req.body = body.to_json
        req.options.on_data = proc do |chunk, _bytes, _env|
          chunks << chunk if chunk
        end
      end

      client.send(:raise_error, response, chunks: chunks) unless response.status == 200
      chunks << response.body if chunks.empty? && response.body.to_s != ""
      chunks
    end

    def persist_capture(chunks, session_id)
      return nil unless @options[:save_capture]

      FileUtils.mkdir_p(DEFAULT_CAPTURE_DIR)
      path = if @options[:save_capture].is_a?(String)
        File.expand_path(@options[:save_capture])
      else
        File.join(DEFAULT_CAPTURE_DIR, "stream-replay-#{session_id}-#{Time.now.strftime('%Y%m%d-%H%M%S')}.sse")
      end
      File.binwrite(path, chunks.join)
      path
    end

    def extract_raw_events(chunks)
      buffer = +""
      events = []

      chunks.each do |chunk|
        temp_client.consume_sse_events_for_debug(chunk, buffer) do |raw_event|
          parsed = parse_sse_json(raw_event)
          choice = parsed&.dig("choices", 0)
          delta = choice&.fetch("delta", {}) || {}
          events << {
            raw_event: raw_event,
            delta_content: delta["content"],
            delta_reasoning_content: delta["reasoning_content"],
            delta_reasoning: delta["reasoning"],
            delta_thinking: delta["thinking"],
            delta_thought: delta["thought"],
            finish_reason: choice&.dig("finish_reason"),
            usage: parsed&.fetch("usage", nil)
          }
        end
      end

      events
    end

    def replay_observer_events(chunks)
      buffer = +""
      events = []

      chunks.each do |chunk|
        temp_client.send(:observe_openai_stream_chunk, chunk, buffer, lambda do |event|
          events << event
        end)
      end

      events
    end

    def parse_sse_json(raw_event)
      data_lines = raw_event.lines.filter_map do |line|
        stripped = line.chomp
        next if stripped.empty? || stripped.start_with?(":")
        next unless stripped.start_with?("data:")

        stripped.sub(/\Adata:\s?/, "")
      end
      return nil if data_lines.empty?

      JSON.parse(data_lines.join("\n"))
    rescue JSON::ParserError
      nil
    end

    def session_summary(session, saved_assistant)
      return nil unless session

      {
        session_id: session[:session_id],
        session_file: @options[:session_file],
        replayed_assistant_index: resolve_assistant_index(session[:messages] || []),
        saved_assistant_content: saved_assistant&.dig(:content),
        saved_assistant_reasoning_content: saved_assistant&.dig(:reasoning_content),
        saved_assistant_content_length: saved_assistant&.dig(:content)&.length || 0,
        saved_assistant_reasoning_length: saved_assistant&.dig(:reasoning_content)&.length || 0,
        config: session[:config]
      }
    end

    def summarize_messages(messages)
      messages.map.with_index do |msg, idx|
        {
          index: idx,
          role: msg[:role],
          content_preview: preview(msg[:content]),
          reasoning_preview: preview(msg[:reasoning_content]),
          system_injected: msg[:system_injected] || false,
          session_context: msg[:session_context] || false
        }
      end
    end

    def summarize_parser_result(result)
      {
        content: result[:content],
        reasoning_content: result[:reasoning_content],
        content_length: result[:content].to_s.length,
        reasoning_length: result[:reasoning_content].to_s.length,
        finish_reason: result[:finish_reason],
        usage: result[:usage]
      }
    end

    def preview(value, limit = 160)
      str = value.to_s
      return str if str.length <= limit

      "#{str[0, limit]}..."
    end

    def temp_client
      @temp_client ||= Client.new("debug", base_url: "https://api.deepseek.com", model: "deepseek-v4-pro")
    end
  end
end

class Clacky::Client
  def consume_sse_events_for_debug(chunk, buffer, &block)
    send(:consume_sse_events, chunk, buffer, &block)
  end
end

options = {
  save_capture: true
}

OptionParser.new do |opts|
  opts.banner = "Usage: rv run ruby scripts/replay_session_stream.rb [options]"

  opts.on("--session ID", "Session id prefix from ~/.clacky/sessions") do |value|
    options[:session] = value
  end

  opts.on("--session-file PATH", "Session json file path") do |value|
    options[:session_file] = value
  end

  opts.on("--assistant-index N", Integer, "Replay messages before this assistant index") do |value|
    options[:assistant_index] = value
  end

  opts.on("--raw-file PATH", "Replay a previously captured SSE file instead of making a live request") do |value|
    options[:raw_file] = value
  end

  opts.on("--save-capture [PATH]", "Save live SSE capture to ~/.clacky/tmp or an explicit path") do |value|
    options[:save_capture] = value || true
  end

  opts.on("--no-save-capture", "Do not persist the live SSE capture") do
    options[:save_capture] = false
  end

  opts.on("--model MODEL", "Override model name for live capture") do |value|
    options[:model] = value
  end

  opts.on("--base-url URL", "Override base URL for live capture") do |value|
    options[:base_url] = value
  end

  opts.on("--max-tokens N", Integer, "Override max_tokens for live capture") do |value|
    options[:max_tokens] = value
  end
end.parse!

Clacky::SessionStreamReplay.new(options).run
