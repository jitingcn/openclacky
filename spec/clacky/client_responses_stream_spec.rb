# frozen_string_literal: true

require "spec_helper"
require "clacky/client"

RSpec.describe Clacky::Client, "Responses API streaming callbacks" do
  let(:api_key) { "sk-openai-test" }
  let(:base_url) { "https://api.openai.com/v1" }
  let(:model) { "gpt-5.1-mini" }

  it "emits reasoning and text deltas while accumulating the final response" do
    usage = double(
      "usage",
      input_tokens: 12,
      output_tokens: 2,
      total_tokens: 14,
      input_tokens_details: nil,
      deep_to_h: { input_tokens: 12, output_tokens: 2, total_tokens: 14 }
    )
    completed_response = double("response", id: "resp_1", usage: usage)
    raw_stream = [
      double("reasoning_delta", type: :"response.reasoning.delta", delta: "Think", deep_to_h: { type: "response.reasoning.delta", delta: "Think" }),
      double("text_delta_1", type: :"response.output_text.delta", delta: "Hello", deep_to_h: { type: "response.output_text.delta", delta: "Hello" }),
      double("text_delta_2", type: :"response.output_text.delta", delta: " world", deep_to_h: { type: "response.output_text.delta", delta: " world" }),
      double("completed", type: :"response.completed", response: completed_response, deep_to_h: { type: "response.completed", response: { id: "resp_1" } })
    ]

    responses_client = double("responses_client")
    allow(responses_client).to receive(:stream_raw).and_return(raw_stream)
    sdk_client = double("sdk_client", responses: responses_client)

    client = described_class.new(
      api_key,
      base_url: base_url,
      model: model,
      stream: true,
      raw_response_logging: true
    )
    client.instance_variable_set(:@use_responses, true)
    client.instance_variable_set(:@openai_sdk_client, sdk_client)

    streamed_events = []
    result = client.send_messages_with_tools(
      [{ role: "user", content: "Say hi" }],
      model: model,
      tools: [],
      max_tokens: 64,
      enable_caching: false,
      on_stream_event: ->(event) { streamed_events << event }
    )

    expect(result[:content]).to eq("Hello world")
    expect(result[:reasoning_content]).to eq("Think")
    expect(result.dig(:raw_response_debug, :transport)).to eq("openai-responses-sdk")
    expect(result.dig(:raw_response_debug, :response_body)).to eq([
      { type: "response.reasoning.delta", delta: "Think" },
      { type: "response.output_text.delta", delta: "Hello" },
      { type: "response.output_text.delta", delta: " world" },
      { type: "response.completed", response: { id: "resp_1" } }
    ])
    expect(result.dig(:latency, :streaming)).to eq(true)
    expect(streamed_events).to eq([
      { content_delta: nil, reasoning_delta: "Think" },
      { content_delta: "Hello", reasoning_delta: nil },
      { content_delta: " world", reasoning_delta: nil }
    ])
  end
end
