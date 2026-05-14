# frozen_string_literal: true

require "spec_helper"
require "clacky/client"
require "faraday"

RSpec.describe Clacky::Client, "Anthropic streaming transport" do
  let(:api_key) { "sk-ant-test" }
  let(:base_url) { "https://api.anthropic.com" }
  let(:model) { "claude-sonnet-4-6" }

  def build_client(connection:, anthropic_stream: true, stream: nil)
    client = described_class.new(
      api_key,
      base_url: base_url,
      model: model,
      anthropic_format: true,
      anthropic_stream: anthropic_stream,
      stream: stream
    )
    client.instance_variable_set(:@anthropic_connection, connection)
    client
  end

  def sse_body(*events)
    events.map do |type, data|
      "event: #{type}\ndata: #{data.to_json}\n\n"
    end.join
  end

  it "routes tool calls through send_anthropic_stream_request and parses SSE chunks" do
    response_body = sse_body(
      ["message_start", { type: "message_start", message: { id: "msg_1", usage: { input_tokens: 12 } } }],
      ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
      ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hello" } }],
      ["content_block_stop", { type: "content_block_stop", index: 0 }],
      ["message_delta", { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 3 } }],
      ["message_stop", { type: "message_stop" }]
    )

    captured_body = nil
    test = Faraday::Adapter::Test::Stubs.new do |stub|
      stub.post("/v1/messages") do |env|
        captured_body = JSON.parse(env.body)
        env.request.on_data.call(response_body, response_body.bytesize, env) if env.request.on_data
        [200, { "Content-Type" => "text/event-stream" }, ""]
      end
    end

    connection = Faraday.new(url: base_url) do |conn|
      conn.adapter :test, test
    end

    client = build_client(connection: connection)
    streamed_events = []
    result = client.send_messages_with_tools(
      [{ role: "user", content: "Say hi" }],
      model: model,
      tools: [],
      max_tokens: 64,
      enable_caching: false,
      on_stream_event: ->(event) { streamed_events << event }
    )

    expect(captured_body["stream"]).to eq(true)
    expect(result[:content]).to eq("Hello")
    expect(result[:finish_reason]).to eq("stop")
    expect(result.dig(:usage, :completion_tokens)).to eq(3)
    expect(result.dig(:latency, :streaming)).to eq(true)
    expect(streamed_events).to include(hash_including(content_delta: "Hello"))

    test.verify_stubbed_calls
  end

  it "routes simple send_messages through the streaming path for anthropic_format clients" do
    response_body = sse_body(
      ["message_start", { type: "message_start", message: { id: "msg_2", usage: { input_tokens: 7 } } }],
      ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
      ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "pong" } }],
      ["content_block_stop", { type: "content_block_stop", index: 0 }],
      ["message_delta", { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } }],
      ["message_stop", { type: "message_stop" }]
    )

    test = Faraday::Adapter::Test::Stubs.new do |stub|
      stub.post("/v1/messages") do |env|
        env.request.on_data.call(response_body, response_body.bytesize, env) if env.request.on_data
        [200, { "Content-Type" => "text/event-stream" }, ""]
      end
    end

    connection = Faraday.new(url: base_url) do |conn|
      conn.adapter :test, test
    end

    client = build_client(connection: connection)
    text = client.send_messages([{ role: "user", content: "ping" }], model: model, max_tokens: 16)

    expect(text).to eq("pong")
    test.verify_stubbed_calls
  end

  it "falls back to non-streaming path when anthropic_stream is false" do
    captured_body = nil
    test = Faraday::Adapter::Test::Stubs.new do |stub|
      stub.post("/v1/messages") do |env|
        captured_body = JSON.parse(env.body)
        [200, { "Content-Type" => "application/json" },
         { content: [{ type: "text", text: "non-streaming response" }],
           stop_reason: "end_turn",
           usage: { input_tokens: 5, output_tokens: 3 } }.to_json]
      end
    end

    connection = Faraday.new(url: base_url) do |conn|
      conn.adapter :test, test
    end

    client = build_client(connection: connection, anthropic_stream: false, stream: false)
    result = client.send_messages_with_tools(
      [{ role: "user", content: "Say hi" }],
      model: model,
      tools: [{ type: "function", function: { name: "test", description: "desc", parameters: {} } }],
      max_tokens: 64,
      enable_caching: false
    )

    expect(captured_body["stream"]).to be_nil
    expect(result[:content]).to eq("non-streaming response")
    expect(result[:finish_reason]).to eq("stop")

    test.verify_stubbed_calls
  end

  it "applies message-level cache_control when caching is enabled" do
    captured_body = nil
    test = Faraday::Adapter::Test::Stubs.new do |stub|
      stub.post("/v1/messages") do |env|
        captured_body = JSON.parse(env.body)
        sse_events = sse_body(
          ["message_start", { type: "message_start", message: { id: "msg_3", usage: { input_tokens: 10 } } }],
          ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
          ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "cached" } }],
          ["content_block_stop", { type: "content_block_stop", index: 0 }],
          ["message_delta", { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } }],
          ["message_stop", { type: "message_stop" }]
        )
        env.request.on_data.call(sse_events, sse_events.bytesize, env) if env.request.on_data
        [200, { "Content-Type" => "text/event-stream" }, ""]
      end
    end

    connection = Faraday.new(url: base_url) do |conn|
      conn.adapter :test, test
    end

    # Use a Claude model that supports caching
    client = described_class.new(
      api_key,
      base_url: base_url,
      model: "claude-sonnet-4-6",
      anthropic_format: true,
      anthropic_stream: true
    )
    client.instance_variable_set(:@anthropic_connection, connection)

    client.send_messages_with_tools(
      [{ role: "user", content: "ping" }, { role: "user", content: "pong" }],
      model: "claude-sonnet-4-6",
      tools: [],
      max_tokens: 64,
      enable_caching: true
    )

    # The LAST message should carry cache_control injected by apply_message_caching
    last_msg = captured_body["messages"].last
    last_block = last_msg["content"].last
    expect(last_block["cache_control"]).to eq({ "type" => "ephemeral" })

    test.verify_stubbed_calls
  end

  it "sends stream: true in anthropic test_connection when anthropic_stream is enabled" do
    captured_body = nil
    test = Faraday::Adapter::Test::Stubs.new do |stub|
      stub.post("/v1/messages") do |env|
        captured_body = JSON.parse(env.body)
        [200, { "Content-Type" => "application/json" }, { id: "msg_test" }.to_json]
      end
    end

    connection = Faraday.new(url: base_url) do |conn|
      conn.adapter :test, test
    end

    client = build_client(connection: connection, anthropic_stream: true)
    result = client.test_connection(model: model)

    expect(result).to eq({ success: true })
    expect(captured_body["stream"]).to eq(true)
    test.verify_stubbed_calls
  end

  it "omits stream in anthropic test_connection when anthropic_stream is disabled" do
    captured_body = nil
    test = Faraday::Adapter::Test::Stubs.new do |stub|
      stub.post("/v1/messages") do |env|
        captured_body = JSON.parse(env.body)
        [200, { "Content-Type" => "application/json" }, { id: "msg_test" }.to_json]
      end
    end

    connection = Faraday.new(url: base_url) do |conn|
      conn.adapter :test, test
    end

    client = build_client(connection: connection, anthropic_stream: false)
    result = client.test_connection(model: model)

    expect(result).to eq({ success: true })
    expect(captured_body.key?("stream")).to eq(false)
    test.verify_stubbed_calls
  end

  it "injects metadata.user_id (cache affinity) in the streaming agent path" do
    response_body = sse_body(
      ["message_start", { type: "message_start", message: { id: "msg_affinity", usage: { input_tokens: 5 } } }],
      ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
      ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "ok" } }],
      ["content_block_stop", { type: "content_block_stop", index: 0 }],
      ["message_delta", { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } }],
      ["message_stop", { type: "message_stop" }]
    )

    captured_body = nil
    test = Faraday::Adapter::Test::Stubs.new do |stub|
      stub.post("/v1/messages") do |env|
        captured_body = JSON.parse(env.body)
        env.request.on_data.call(response_body, response_body.bytesize, env) if env.request.on_data
        [200, { "Content-Type" => "text/event-stream" }, ""]
      end
    end

    connection = Faraday.new(url: base_url) do |conn|
      conn.adapter :test, test
    end

    client = build_client(connection: connection)
    client.send_messages_with_tools(
      [{ role: "user", content: "hi" }],
      model: model,
      tools: [],
      max_tokens: 64,
      enable_caching: false
    )

    # Verify cache affinity metadata is injected in the streaming request body
    expect(captured_body["metadata"]).to be_a(Hash)
    user_id_json = captured_body["metadata"]["user_id"]
    parsed = JSON.parse(user_id_json)
    expect(parsed).to include("device_id", "account_uuid", "session_id")
    expect(parsed["session_id"]).to eq(client.cache_affinity_session_id)

    test.verify_stubbed_calls
  end

  it "injects metadata.user_id (cache affinity) in the simple streaming path" do
    response_body = sse_body(
      ["message_start", { type: "message_start", message: { id: "msg_simple_affinity", usage: { input_tokens: 3 } } }],
      ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
      ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "pong" } }],
      ["content_block_stop", { type: "content_block_stop", index: 0 }],
      ["message_delta", { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } }],
      ["message_stop", { type: "message_stop" }]
    )

    captured_body = nil
    test = Faraday::Adapter::Test::Stubs.new do |stub|
      stub.post("/v1/messages") do |env|
        captured_body = JSON.parse(env.body)
        env.request.on_data.call(response_body, response_body.bytesize, env) if env.request.on_data
        [200, { "Content-Type" => "text/event-stream" }, ""]
      end
    end

    connection = Faraday.new(url: base_url) do |conn|
      conn.adapter :test, test
    end

    client = build_client(connection: connection)
    client.send_messages([{ role: "user", content: "ping" }], model: model, max_tokens: 16)

    # Verify cache affinity metadata is injected in the simple streaming path
    expect(captured_body["metadata"]).to be_a(Hash)
    user_id_json = captured_body["metadata"]["user_id"]
    parsed = JSON.parse(user_id_json)
    expect(parsed["session_id"]).to eq(client.cache_affinity_session_id)

    test.verify_stubbed_calls
  end
end
