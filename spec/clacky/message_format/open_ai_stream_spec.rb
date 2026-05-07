# frozen_string_literal: true

require "spec_helper"
require "json"

RSpec.describe Clacky::MessageFormat::OpenAI, "streaming" do
  # ── parse_stream_response ──────────────────────────────────────────────────

  describe ".parse_stream_response" do
    # Helper: build SSE chunks from an array of choice hashes
    def build_sse_chunks(deltas, usage: nil, model: "gpt-4")
      chunks = []
      deltas.each_with_index do |delta, i|
        data = {
          "id" => "chatcmpl-test#{i}",
          "object" => "chat.completion.chunk",
          "created" => 1_778_159_645,
          "model" => model,
          "choices" => [
            { "delta" => delta, "logprobs" => nil, "finish_reason" => nil, "index" => 0 }
          ],
          "usage" => nil
        }
        chunks << "data: #{data.to_json}\n\n"
      end

      # Final chunk with finish_reason
      last_delta = deltas.last || {}
      finish_reason = last_delta.key?("tool_calls") ? "tool_calls" : "stop"
      final_data = {
        "id" => "chatcmpl-test-final",
        "object" => "chat.completion.chunk",
        "created" => 1_778_159_645,
        "model" => model,
        "choices" => [
          { "delta" => {}, "logprobs" => nil, "finish_reason" => finish_reason, "index" => 0 }
        ],
        "usage" => nil
      }
      chunks << "data: #{final_data.to_json}\n\n"

      # Usage chunk (empty choices)
      if usage
        usage_data = {
          "id" => "chatcmpl-test-usage",
          "object" => "chat.completion.chunk",
          "created" => 1_778_159_645,
          "model" => model,
          "choices" => [],
          "usage" => usage
        }
        chunks << "data: #{usage_data.to_json}\n\n"
      end

      chunks << "data: [DONE]\n\n"
      chunks
    end

    it "parses a simple text streaming response" do
      chunks = build_sse_chunks([
        { "role" => "assistant", "content" => "" },
        { "content" => "Hello" },
        { "content" => " world" }
      ], usage: { "prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15 })

      result = described_class.parse_stream_response(chunks)

      expect(result[:content]).to eq("Hello world")
      expect(result[:tool_calls]).to be_nil
      expect(result[:finish_reason]).to eq("stop")
      expect(result[:usage][:prompt_tokens]).to eq(10)
      expect(result[:usage][:completion_tokens]).to eq(5)
    end

    it "parses a single tool call streaming response" do
      chunks = build_sse_chunks([
        { "role" => "assistant", "content" => "" },
        { "tool_calls" => [{ "index" => 0, "id" => "call_abc123", "type" => "function", "function" => { "name" => "terminal", "arguments" => "" } }] },
        { "tool_calls" => [{ "index" => 0, "id" => "call_abc123", "type" => "function", "function" => { "name" => "terminal", "arguments" => "{\"command\":\"echo hello\"}" } }] },
        { "tool_calls" => [{ "index" => 0, "id" => "call_abc123", "type" => "function", "function" => { "name" => "terminal", "arguments" => "" } }] }
      ], usage: { "prompt_tokens" => 50, "completion_tokens" => 20, "total_tokens" => 70 })

      result = described_class.parse_stream_response(chunks)

      expect(result[:tool_calls]).to be_an(Array)
      expect(result[:tool_calls].length).to eq(1)
      expect(result[:tool_calls][0][:id]).to eq("call_abc123")
      expect(result[:tool_calls][0][:name]).to eq("terminal")
      expect(JSON.parse(result[:tool_calls][0][:arguments])).to eq({ "command" => "echo hello" })
      expect(result[:finish_reason]).to eq("tool_calls")
    end

    it "parses multiple tool calls without infinite loop (regression guard)" do
      # This test guards against the bug where `idx` was overwritten by `tc["index"]`
      # causing an infinite loop when tool_calls with index > 0 were present.
      chunks = build_sse_chunks([
        { "role" => "assistant", "content" => "" },
        { "tool_calls" => [{ "index" => 0, "id" => "call_001", "type" => "function", "function" => { "name" => "terminal", "arguments" => "" } }] },
        { "tool_calls" => [{ "index" => 0, "id" => "call_001", "type" => "function", "function" => { "name" => "terminal", "arguments" => "{\"command\":\"echo a\"}" } }] },
        { "tool_calls" => [{ "index" => 0, "id" => "call_001", "type" => "function", "function" => { "name" => "terminal", "arguments" => "" } }] },
        { "tool_calls" => [{ "index" => 1, "id" => "call_002", "type" => "function", "function" => { "name" => "terminal", "arguments" => "" } }] },
        { "tool_calls" => [{ "index" => 1, "id" => "call_002", "type" => "function", "function" => { "name" => "terminal", "arguments" => "{\"command\":\"echo b\"}" } }] },
        { "tool_calls" => [{ "index" => 1, "id" => "call_002", "type" => "function", "function" => { "name" => "terminal", "arguments" => "" } }] }
      ], usage: { "prompt_tokens" => 100, "completion_tokens" => 40, "total_tokens" => 140 })

      require "timeout"
      result = Timeout.timeout(5) { described_class.parse_stream_response(chunks) }

      expect(result[:tool_calls]).to be_an(Array)
      expect(result[:tool_calls].length).to eq(2)
      expect(result[:tool_calls][0][:name]).to eq("terminal")
      expect(result[:tool_calls][0][:id]).to eq("call_001")
      expect(result[:tool_calls][1][:name]).to eq("terminal")
      expect(result[:tool_calls][1][:id]).to eq("call_002")
      expect(JSON.parse(result[:tool_calls][0][:arguments])).to eq({ "command" => "echo a" })
      expect(JSON.parse(result[:tool_calls][1][:arguments])).to eq({ "command" => "echo b" })
    end

    it "does not duplicate tool call id/name when repeated across chunks" do
      chunks = build_sse_chunks([
        { "role" => "assistant", "content" => "" },
        { "tool_calls" => [{ "index" => 0, "id" => "call_repeat", "type" => "function", "function" => { "name" => "test_fn", "arguments" => "" } }] },
        { "tool_calls" => [{ "index" => 0, "id" => "call_repeat", "type" => "function", "function" => { "name" => "test_fn", "arguments" => "{\"a\":1}" } }] },
        { "tool_calls" => [{ "index" => 0, "id" => "call_repeat", "type" => "function", "function" => { "name" => "test_fn", "arguments" => "" } }] }
      ])

      result = described_class.parse_stream_response(chunks)

      expect(result[:tool_calls][0][:id]).to eq("call_repeat")
      expect(result[:tool_calls][0][:id].length).to eq("call_repeat".length)
      expect(result[:tool_calls][0][:name]).to eq("test_fn")
    end

    it "handles empty content response" do
      chunks = build_sse_chunks([
        { "role" => "assistant", "content" => "" }
      ])

      result = described_class.parse_stream_response(chunks)

      expect(result[:content]).to be_nil
      expect(result[:finish_reason]).to eq("stop")
    end

    it "handles chunks split across SSE boundaries" do
      # Simulate Faraday splitting a single JSON line across two chunks
      full_json = {
        "id" => "chatcmpl-split",
        "object" => "chat.completion.chunk",
        "created" => 1_778_159_645,
        "model" => "gpt-4",
        "choices" => [
          { "delta" => { "content" => "Hi" }, "logprobs" => nil, "finish_reason" => nil, "index" => 0 }
        ],
        "usage" => nil
      }.to_json

      # Split the JSON in the middle
      mid = full_json.length / 2
      chunks = [
        "data: #{full_json[0...mid]}\n",
        "#{full_json[mid..]}\n\n",
        "data: [DONE]\n\n"
      ]

      result = described_class.parse_stream_response(chunks)

      expect(result[:content]).to eq("Hi")
    end
  end
end
