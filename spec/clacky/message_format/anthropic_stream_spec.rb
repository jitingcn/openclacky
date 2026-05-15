# frozen_string_literal: true

require "spec_helper"
require "json"

RSpec.describe Clacky::MessageFormat::Anthropic do
  describe ".build_request_body" do
    it "adds adaptive thinking and output_config for Claude 4.6+" do
      body = described_class.build_request_body(
        [{ role: "user", content: "hello" }],
        "claude-sonnet-4.6",
        [],
        4096,
        false,
        thinking_enabled: true,
        reasoning_effort: "max",
        provider_id: "anthropic"
      )

      expect(body[:thinking]).to eq({ type: "adaptive" })
      expect(body[:output_config]).to eq({ effort: "high" })
    end

    it "round-trips stored thinking blocks ahead of tool_use blocks" do
      body = described_class.build_request_body(
        [{
          role: "assistant",
          content: "",
          thinking_blocks: [
            { type: "thinking", thinking: "Need a tool.", signature: "sig_123" }
          ],
          tool_calls: [
            { id: "call_1", type: "function", name: "read_file", arguments: '{"path":"README.md"}' }
          ]
        }],
        "claude-sonnet-4.6",
        [],
        4096,
        false
      )

      blocks = body.fetch(:messages).first.fetch(:content)
      expect(blocks.first).to eq({ type: "thinking", thinking: "Need a tool.", signature: "sig_123" })
      expect(blocks.last[:type]).to eq("tool_use")
      expect(blocks.last[:id]).to eq("call_1")
    end
  end

  describe ".parse_response" do
    it "preserves thinking blocks and exposes text for UI rendering" do
      data = {
        "content" => [
          { "type" => "thinking", "thinking" => "Let me think.", "signature" => "sig_1" },
          { "type" => "text", "text" => "Done." }
        ],
        "usage" => { "input_tokens" => 10, "output_tokens" => 2 }
      }

      result = described_class.parse_response(data)
      expect(result[:thinking]).to eq("Let me think.")
      expect(result[:thinking_blocks]).to eq([
        { type: "thinking", thinking: "Let me think.", signature: "sig_1" }
      ])
      expect(result[:content]).to eq("Done.")
    end
  end

  describe ".format_tool_results" do
    it "accepts tool_call_id/result shaped tool results" do
      response = {
        tool_calls: [
          { id: "call_123", type: "function", name: "echo_text", arguments: '{"text":"pong"}' }
        ]
      }
      tool_results = [
        { tool_call_id: "call_123", tool_name: "echo_text", result: { text: "pong" } }
      ]

      result = described_class.format_tool_results(response, tool_results)

      expect(result).to eq([
        {
          role: "tool",
          tool_call_id: "call_123",
          content: '{"text":"pong"}'
        }
      ])
    end

    it "keeps existing id/content shaped tool results working" do
      response = {
        tool_calls: [
          { id: "call_456", type: "function", name: "echo_text", arguments: '{"text":"pong"}' }
        ]
      }
      tool_results = [
        { id: "call_456", content: "pong" }
      ]

      result = described_class.format_tool_results(response, tool_results)

      expect(result).to eq([
        {
          role: "tool",
          tool_call_id: "call_456",
          content: "pong"
        }
      ])
    end
  end
end

RSpec.describe Clacky::MessageFormat::Anthropic, "streaming" do
  # ── build_stream_request_body ──────────────────────────────────────────────

  describe ".build_stream_request_body" do
    it "includes stream: true in the body" do
      body = described_class.build_stream_request_body(
        [{ role: "user", content: "hello" }],
        "claude-sonnet-4-6",
        [],
        1024,
        false
      )
      expect(body[:stream]).to eq(true)
    end

    it "preserves all fields from build_request_body" do
      tools = [{ type: "function", function: { name: "test", description: "desc", parameters: {} } }]
      body = described_class.build_stream_request_body(
        [{ role: "user", content: "hello" }],
        "claude-sonnet-4-6",
        tools,
        2048,
        true
      )
      expect(body[:model]).to eq("claude-sonnet-4-6")
      expect(body[:max_tokens]).to eq(2048)
      expect(body[:messages]).to be_an(Array)
      expect(body[:tools]).to be_an(Array)
      # Caching: last tool should have cache_control
      expect(body[:tools].last[:cache_control]).to eq({ type: "ephemeral" })
    end
  end

  # ── parse_stream_response ──────────────────────────────────────────────────

  describe ".parse_stream_response" do
    # Helper: build SSE string from event type + JSON data pairs
    def build_sse(events)
      events.map do |type, data|
        "event: #{type}\ndata: #{data.to_json}\n\n"
      end.join
    end

    it "parses a simple text streaming response" do
      sse = build_sse([
        ["message_start", { type: "message_start", message: { id: "msg_1", usage: { input_tokens: 50 } } }],
        ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hello" } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: " world" } }],
        ["content_block_stop", { type: "content_block_stop", index: 0 }],
        ["message_delta", { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 5 } }],
        ["message_stop", { type: "message_stop" }]
      ])

      result = described_class.parse_stream_response([sse])

      expect(result[:content]).to eq("Hello world")
      expect(result[:tool_calls]).to be_nil
      expect(result[:finish_reason]).to eq("stop")
      expect(result[:usage][:completion_tokens]).to eq(5)
      expect(result[:usage][:prompt_tokens]).to eq(50)
    end

    it "parses a tool_use streaming response" do
      sse = build_sse([
        ["message_start", { type: "message_start", message: { id: "msg_2", usage: { input_tokens: 100 } } }],
        ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Let me check" } }],
        ["content_block_stop", { type: "content_block_stop", index: 0 }],
        ["content_block_start", { type: "content_block_start", index: 1, content_block: { type: "tool_use", id: "toolu_abc", name: "get_weather", input: {} } }],
        ["content_block_delta", { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: '{"ci' } }],
        ["content_block_delta", { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: 'ty":"SF"}' } }],
        ["content_block_stop", { type: "content_block_stop", index: 1 }],
        ["message_delta", { type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 30 } }],
        ["message_stop", { type: "message_stop" }]
      ])

      result = described_class.parse_stream_response([sse])

      expect(result[:content]).to eq("Let me check")
      expect(result[:tool_calls]).to be_an(Array)
      expect(result[:tool_calls].length).to eq(1)
      expect(result[:tool_calls][0][:id]).to eq("toolu_abc")
      expect(result[:tool_calls][0][:name]).to eq("get_weather")
      expect(JSON.parse(result[:tool_calls][0][:arguments])).to eq({ "city" => "SF" })
      expect(result[:finish_reason]).to eq("tool_calls")
    end

    it "handles cache_read_input_tokens from streaming" do
      sse = build_sse([
        ["message_start", { type: "message_start", message: { id: "msg_3", usage: { input_tokens: 10, cache_read_input_tokens: 500 } } }],
        ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Cached" } }],
        ["content_block_stop", { type: "content_block_stop", index: 0 }],
        ["message_delta", { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1, cache_read_input_tokens: 100 } }],
        ["message_stop", { type: "message_stop" }]
      ])

      result = described_class.parse_stream_response([sse])

      # Anthropic usually reports cache_read only once; if a delta repeats the
      # field, the parser should prefer the latest value rather than double-count.
      expect(result[:usage][:cache_read_input_tokens]).to eq(100)
      # prompt_tokens = raw_input(10) + cache_read(100) = 110
      expect(result[:usage][:prompt_tokens]).to eq(110)
    end

    it "returns nil content when no text deltas arrive" do
      sse = build_sse([
        ["message_start", { type: "message_start", message: { id: "msg_4", usage: { input_tokens: 20 } } }],
        ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "tool_use", id: "toolu_x", name: "act", input: {} } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: "{}" } }],
        ["content_block_stop", { type: "content_block_stop", index: 0 }],
        ["message_delta", { type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 10 } }],
        ["message_stop", { type: "message_stop" }]
      ])

      result = described_class.parse_stream_response([sse])

      expect(result[:content]).to be_nil
      expect(result[:tool_calls].length).to eq(1)
      expect(result[:finish_reason]).to eq("tool_calls")
    end

    it "handles multiple tool calls in a single stream" do
      sse = build_sse([
        ["message_start", { type: "message_start", message: { id: "msg_5", usage: { input_tokens: 30 } } }],
        ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "tool_use", id: "toolu_a", name: "tool_a", input: {} } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: '{"x":1}' } }],
        ["content_block_stop", { type: "content_block_stop", index: 0 }],
        ["content_block_start", { type: "content_block_start", index: 1, content_block: { type: "tool_use", id: "toolu_b", name: "tool_b", input: {} } }],
        ["content_block_delta", { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: '{"y":2}' } }],
        ["content_block_stop", { type: "content_block_stop", index: 1 }],
        ["message_delta", { type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 20 } }],
        ["message_stop", { type: "message_stop" }]
      ])

      result = described_class.parse_stream_response([sse])

      expect(result[:tool_calls].length).to eq(2)
      expect(result[:tool_calls][0][:name]).to eq("tool_a")
      expect(result[:tool_calls][1][:name]).to eq("tool_b")
    end

    it "uses input_tokens from message_delta when providers report final usage there" do
      sse = build_sse([
        ["message_start", { type: "message_start", message: { id: "msg_7", usage: { input_tokens: 0, output_tokens: 0 } } }],
        ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Mimi" } }],
        ["content_block_stop", { type: "content_block_stop", index: 0 }],
        ["message_delta", { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { input_tokens: 29, output_tokens: 3, cache_read_input_tokens: 0 } }],
        ["message_stop", { type: "message_stop" }]
      ])

      result = described_class.parse_stream_response([sse])

      expect(result[:content]).to eq("Mimi")
      expect(result[:usage][:prompt_tokens]).to eq(29)
      expect(result[:usage][:completion_tokens]).to eq(3)
      expect(result[:usage][:total_tokens]).to eq(32)
    end

    it "captures thinking deltas without losing later text output" do
      sse = build_sse([
        ["message_start", { type: "message_start", message: { id: "msg_8", usage: { input_tokens: 20 } } }],
        ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "" } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "Let me think. " } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "Checking context." } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "signature_delta", signature: "sig_abc" } }],
        ["content_block_stop", { type: "content_block_stop", index: 0 }],
        ["content_block_start", { type: "content_block_start", index: 1, content_block: { type: "text", text: "" } }],
        ["content_block_delta", { type: "content_block_delta", index: 1, delta: { type: "text_delta", text: "Mimi" } }],
        ["content_block_stop", { type: "content_block_stop", index: 1 }],
        ["message_delta", { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 3 } }],
        ["message_stop", { type: "message_stop" }]
      ])

      result = described_class.parse_stream_response([sse])

      expect(result[:thinking]).to eq("Let me think. Checking context.")
      expect(result[:thinking_blocks]).to eq([
        { type: "thinking", thinking: "Let me think. Checking context.", signature: "sig_abc" }
      ])
      expect(result[:content]).to eq("Mimi")
      expect(result[:finish_reason]).to eq("stop")
    end

    it "captures thinking-only responses without fabricating text content" do
      sse = build_sse([
        ["message_start", { type: "message_start", message: { id: "msg_9", usage: { input_tokens: 12 } } }],
        ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "" } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "Internal reasoning" } }],
        ["content_block_stop", { type: "content_block_stop", index: 0 }],
        ["message_delta", { type: "message_delta", delta: { stop_reason: "max_tokens" }, usage: { output_tokens: 5 } }],
        ["message_stop", { type: "message_stop" }]
      ])

      result = described_class.parse_stream_response([sse])

      expect(result[:thinking]).to eq("Internal reasoning")
      expect(result[:content]).to be_nil
      expect(result[:finish_reason]).to eq("length")
    end

    it "handles SSE without event: lines (data-only fallback)" do
      # Some providers may send data: lines without event: prefixes
      raw = [
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_6\",\"usage\":{\"input_tokens\":10}}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n"
      ]

      result = described_class.parse_stream_response(raw)

      expect(result[:content]).to eq("Hi")
      expect(result[:finish_reason]).to eq("stop")
    end
  end
end
