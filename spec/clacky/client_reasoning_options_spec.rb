# frozen_string_literal: true

require "spec_helper"

RSpec.describe Clacky::Client, "thinking/reasoning request options" do
  it "adds DeepSeek openai-compatible thinking toggle and disables reasoning when turned off" do
    client = described_class.new(
      "sk-test",
      base_url: "https://api.deepseek.com",
      model: "deepseek-v4-pro",
      thinking_enabled: false,
      reasoning_effort: "high"
    )

    body = client.send(:build_openai_request_body, [{ role: "user", content: "hi" }], "deepseek-v4-pro", [], 128, false)

    expect(body[:thinking]).to eq({ type: "disabled" })
    expect(body[:reasoning_effort]).to eq("none")
  end

  it "adds adaptive thinking and normalized effort on anthropic-native Claude 4.6+" do
    client = described_class.new(
      "sk-ant",
      base_url: "https://api.anthropic.com",
      model: "claude-sonnet-4.6",
      anthropic_format: true,
      thinking_enabled: true,
      reasoning_effort: "max"
    )

    body = client.send(:build_anthropic_request_body, [{ role: "user", content: "hi" }], "claude-sonnet-4.6", [], 4096, false)

    expect(body[:thinking]).to eq({ type: "adaptive" })
    expect(body[:output_config]).to eq({ effort: "high" })
  end

  it "passes xhigh through on the OpenAI-compatible path when explicitly selected" do
    client = described_class.new(
      "sk-test",
      base_url: "https://api.openai.com/v1",
      model: "gpt-5.2",
      reasoning_effort: "xhigh"
    )

    body = client.send(:build_openai_request_body, [{ role: "user", content: "hi" }], "gpt-5.2", [], 128, false)

    expect(body[:reasoning_effort]).to eq("xhigh")
  end
end
