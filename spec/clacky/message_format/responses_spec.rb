# frozen_string_literal: true

require "spec_helper"

RSpec.describe Clacky::MessageFormat::Responses do
  describe ".build_request_body" do
    it "adds reasoning options when thinking_level is enabled" do
      body = described_class.build_request_body(
        [{ role: "user", content: "hi" }],
        "gpt-5.4",
        [],
        128,
        false,
        thinking_level: "medium"
      )

      expect(body[:reasoning]).to eq({ effort: "medium" })
    end
  end

  describe ".parse_response" do
    it "extracts reasoning text from assistant message content" do
      data = {
        "output" => [
          {
            "type" => "message",
            "role" => "assistant",
            "content" => [
              { "type" => "reasoning", "text" => "先思考。" },
              { "type" => "output_text", "text" => "2" }
            ]
          }
        ],
        "usage" => {
          "input_tokens" => 4,
          "output_tokens" => 6,
          "total_tokens" => 10
        }
      }

      result = described_class.parse_response(data)
      expect(result[:content]).to eq("2")
      expect(result[:reasoning_content]).to eq("先思考。")
    end
  end

  describe ".parse_stream_response" do
    it "extracts reasoning delta events" do
      raw_chunks = [
        "event: response.reasoning.delta\n",
        "data: {\"delta\":\"先\"}\n\n",
        "event: response.reasoning_summary_text.delta\n",
        "data: {\"delta\":\"想\"}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"delta\":\"2\"}\n\n",
        "event: response.completed\n",
        "data: {\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":4,\"output_tokens\":6,\"total_tokens\":10}}}\n\n"
      ]

      result = described_class.parse_stream_response(raw_chunks)
      expect(result[:content]).to eq("2")
      expect(result[:reasoning_content]).to eq("先想")
    end
  end
end
