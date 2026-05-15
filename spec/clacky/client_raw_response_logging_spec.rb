# frozen_string_literal: true

require "spec_helper"
require "clacky/client"
require "faraday"

RSpec.describe Clacky::Client, "raw response logging" do
  let(:api_key) { "sk-openai-test" }
  let(:base_url) { "https://api.openai.com/v1" }
  let(:model) { "gpt-5.2" }

  it "captures raw OpenAI-compatible SSE bodies when enabled" do
    response_body = [
      %(data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"finish_reason":null,"index":0}],"usage":null}\n\n),
      %(data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":null}\n\n),
      %(data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\n),
      "data: [DONE]\n\n"
    ].join

    test = Faraday::Adapter::Test::Stubs.new do |stub|
      stub.post("/v1/chat/completions") do |env|
        env.request.on_data.call(response_body, response_body.bytesize, env) if env.request.on_data
        [200, { "Content-Type" => "text/event-stream" }, ""]
      end
    end

    connection = Faraday.new(url: base_url) do |conn|
      conn.adapter :test, test
    end

    client = described_class.new(
      api_key,
      base_url: base_url,
      model: model,
      stream: true,
      raw_response_logging: true
    )
    client.instance_variable_set(:@openai_connection, connection)

    result = client.send_messages_with_tools(
      [{ role: "user", content: "Say hi" }],
      model: model,
      tools: [],
      max_tokens: 64,
      enable_caching: false
    )

    expect(result[:content]).to eq("Hello")
    expect(result.dig(:raw_response_debug, :transport)).to eq("openai-chat-completions")
    expect(result.dig(:raw_response_debug, :streaming)).to eq(true)
    expect(result.dig(:raw_response_debug, :content_type)).to eq("text/event-stream")
    expect(result.dig(:raw_response_debug, :response_body)).to eq(response_body)
    test.verify_stubbed_calls
  end
end
