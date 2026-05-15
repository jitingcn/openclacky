# frozen_string_literal: true

require "spec_helper"

RSpec.describe Clacky::MessageFormat::OpenAI do
  describe ".build_request_body" do
    let(:model) { "deepseek-v4-pro" }
    let(:tools) { [] }
    let(:max_tokens) { 1024 }

    it "passes through plain text messages unchanged" do
      messages = [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there!" }
      ]

      body = described_class.build_request_body(messages, model, tools, max_tokens, false)
      expect(body[:messages]).to eq(messages)
    end

    it "passes through text-only content arrays unchanged" do
      messages = [
        { role: "user", content: [{ type: "text", text: "Hello" }] }
      ]

      body = described_class.build_request_body(messages, model, tools, max_tokens, false)
      expect(body[:messages]).to eq(messages)
    end

    it "keeps image_url blocks when vision_supported is true (default)" do
      messages = [
        { role: "user", content: [
          { type: "text", text: "Look at this:" },
          { type: "image_url", image_url: { url: "data:image/png;base64,abc123" } }
        ] }
      ]

      body = described_class.build_request_body(messages, model, tools, max_tokens, false)
      expect(body[:messages].first[:content].length).to eq(2)
      expect(body[:messages].first[:content][1][:type]).to eq("image_url")
    end

    it "converts image_url blocks to text placeholders when vision_supported is false" do
      messages = [
        { role: "user", content: [
          { type: "text", text: "Look at this:" },
          { type: "image_url", image_url: { url: "data:image/png;base64,abc123" } }
        ] }
      ]

      body = described_class.build_request_body(
        messages, model, tools, max_tokens, false,
        vision_supported: false
      )
      result_content = body[:messages].first[:content]
      # Both blocks remain: the original text + image_url replaced with text placeholder
      expect(result_content.length).to eq(2)
      expect(result_content[0][:type]).to eq("text")
      expect(result_content[0][:text]).to eq("Look at this:")
      expect(result_content[1][:type]).to eq("text")
      expect(result_content[1][:text]).to include("Image content removed")
    end

    it "replaces a sole image_url block with a placeholder text when vision_supported is false" do
      messages = [
        { role: "user", content: [
          { type: "image_url", image_url: { url: "data:image/png;base64,abc123" } }
        ] }
      ]

      body = described_class.build_request_body(
        messages, model, tools, max_tokens, false,
        vision_supported: false
      )
      result_content = body[:messages].first[:content]
      expect(result_content.length).to eq(1)
      expect(result_content.first[:type]).to eq("text")
      expect(result_content.first[:text]).to include("Image content removed")
    end

    it "drops empty text blocks during conversion" do
      messages = [
        { role: "user", content: [
          { type: "text", text: "" },
          { type: "text", text: "Valid text" }
        ] }
      ]

      body = described_class.build_request_body(messages, model, tools, max_tokens, false)
      result_content = body[:messages].first[:content]
      expect(result_content.length).to eq(1)
      expect(result_content.first[:text]).to eq("Valid text")
    end

    it "preserves cache_control on text blocks" do
      messages = [
        { role: "user", content: [
          { type: "text", text: "Cached text", cache_control: { type: "ephemeral" } }
        ] }
      ]

      body = described_class.build_request_body(messages, model, tools, max_tokens, false)
      result_content = body[:messages].first[:content]
      expect(result_content.first[:cache_control]).to eq({ type: "ephemeral" })
    end

    it "handles messages with String content (no conversion needed)" do
      messages = [
        { role: "user", content: "Plain string content" },
        { role: "assistant", content: "Another string" }
      ]

      body = described_class.build_request_body(
        messages, model, tools, max_tokens, false,
        vision_supported: false
      )
      expect(body[:messages].first[:content]).to eq("Plain string content")
      expect(body[:messages].last[:content]).to eq("Another string")
    end

    it "adds reasoning_effort when configured" do
      messages = [
        { role: "user", content: "Hello" }
      ]

      body = described_class.build_request_body(
        messages, model, tools, max_tokens, false,
        reasoning_effort: "medium"
      )
      expect(body[:reasoning_effort]).to eq("medium")
      expect(body).not_to have_key(:reasoning)
    end

    it "parses reasoning from non-streaming responses" do
      data = {
        "choices" => [
          {
            "message" => {
              "role" => "assistant",
              "content" => "2",
              "reasoning" => "先简单思考。"
            },
            "finish_reason" => "stop"
          }
        ],
        "usage" => {
          "prompt_tokens" => 3,
          "completion_tokens" => 5,
          "total_tokens" => 8
        }
      }

      result = described_class.parse_response(data)
      expect(result[:content]).to eq("2")
      expect(result[:reasoning_content]).to eq("先简单思考。")
    end

    it "parses reasoning from thinking field (alternative vendor convention)" do
      data = {
        "choices" => [
          {
            "message" => {
              "role" => "assistant",
              "content" => "42",
              "thinking" => "Let me calculate..."
            },
            "finish_reason" => "stop"
          }
        ],
        "usage" => { "prompt_tokens" => 1, "completion_tokens" => 2, "total_tokens" => 3 }
      }

      result = described_class.parse_response(data)
      expect(result[:content]).to eq("42")
      expect(result[:reasoning_content]).to eq("Let me calculate...")
    end

    it "parses reasoning from thought field (Chinese vendor convention)" do
      data = {
        "choices" => [
          {
            "message" => {
              "role" => "assistant",
              "content" => "99",
              "thought" => "这很简单"
            },
            "finish_reason" => "stop"
          }
        ],
        "usage" => { "prompt_tokens" => 1, "completion_tokens" => 2, "total_tokens" => 3 }
      }

      result = described_class.parse_response(data)
      expect(result[:content]).to eq("99")
      expect(result[:reasoning_content]).to eq("这很简单")
    end

    it "parses leading <think> block from content as reasoning_content" do
      data = {
        "choices" => [
          {
            "message" => {
              "role" => "assistant",
              "content" => "<think>先想一下</think>\n\n最终答案"
            },
            "finish_reason" => "stop"
          }
        ],
        "usage" => { "prompt_tokens" => 1, "completion_tokens" => 2, "total_tokens" => 3 }
      }

      result = described_class.parse_response(data)
      expect(result[:reasoning_content]).to eq("先想一下")
      expect(result[:content]).to eq("最终答案")
    end

    it "parses leading <thinking> block from content as reasoning_content" do
      data = {
        "choices" => [
          {
            "message" => {
              "role" => "assistant",
              "content" => "\n  <thinking>multi\nline\nthought</thinking>\nAnswer"
            },
            "finish_reason" => "stop"
          }
        ],
        "usage" => { "prompt_tokens" => 1, "completion_tokens" => 2, "total_tokens" => 3 }
      }

      result = described_class.parse_response(data)
      expect(result[:reasoning_content]).to eq("multi\nline\nthought")
      expect(result[:content]).to eq("Answer")
    end

    it "does not extract think tags that are not at the beginning of content" do
      data = {
        "choices" => [
          {
            "message" => {
              "role" => "assistant",
              "content" => "前言\n<think>中间的想法</think>\n结尾"
            },
            "finish_reason" => "stop"
          }
        ],
        "usage" => { "prompt_tokens" => 1, "completion_tokens" => 2, "total_tokens" => 3 }
      }

      result = described_class.parse_response(data)
      expect(result[:reasoning_content]).to be_nil
      expect(result[:content]).to eq("前言\n<think>中间的想法</think>\n结尾")
    end

    it "prefers explicit reasoning fields over extracted leading think block" do
      data = {
        "choices" => [
          {
            "message" => {
              "role" => "assistant",
              "content" => "<think>内联思考</think>\n答案",
              "reasoning_content" => "显式思考"
            },
            "finish_reason" => "stop"
          }
        ],
        "usage" => { "prompt_tokens" => 1, "completion_tokens" => 2, "total_tokens" => 3 }
      }

      result = described_class.parse_response(data)
      expect(result[:reasoning_content]).to eq("显式思考")
      expect(result[:content]).to eq("答案")
    end

    it "parses reasoning from streaming deltas" do
      raw_chunks = [
        "data: {\"choices\":[{\"delta\":{\"reasoning\":\"先\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"想\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"2\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":5,\"total_tokens\":8}}\n\n",
        "data: [DONE]\n\n"
      ]

      result = described_class.parse_stream_response(raw_chunks)
      expect(result[:content]).to eq("2")
      expect(result[:reasoning_content]).to eq("先想")
    end

    it "parses reasoning from thinking delta (alternative vendor convention)" do
      raw_chunks = [
        "data: {\"choices\":[{\"delta\":{\"thinking\":\"Let me\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"thinking\":\" think\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"42\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n",
        "data: [DONE]\n\n"
      ]

      result = described_class.parse_stream_response(raw_chunks)
      expect(result[:content]).to eq("42")
      expect(result[:reasoning_content]).to eq("Let me think")
    end

    it "parses reasoning from thought delta (Chinese vendor convention)" do
      raw_chunks = [
        "data: {\"choices\":[{\"delta\":{\"thought\":\"一步\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"thought\":\"一步来\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"99\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n",
        "data: [DONE]\n\n"
      ]

      result = described_class.parse_stream_response(raw_chunks)
      expect(result[:content]).to eq("99")
      expect(result[:reasoning_content]).to eq("一步一步来")
    end

    it "handles mixed content with multiple image_url blocks when vision_supported is false" do
      messages = [
        { role: "user", content: [
          { type: "image_url", image_url: { url: "data:image/png;base64,img1" } },
          { type: "text", text: "Between images" },
          { type: "image_url", image_url: { url: "data:image/png;base64,img2" } }
        ] }
      ]

      body = described_class.build_request_body(
        messages, model, tools, max_tokens, false,
        vision_supported: false
      )
      result_content = body[:messages].first[:content]
      # All 3 blocks remain, but image_url blocks become text placeholders
      expect(result_content.length).to eq(3)
      expect(result_content[0][:text]).to include("Image content removed")
      expect(result_content[1][:text]).to eq("Between images")
      expect(result_content[2][:text]).to include("Image content removed")
    end
  end

  describe ".normalize_block" do
    it "returns nil for empty text blocks" do
      result = described_class.normalize_block(
        { type: "text", text: "" },
        vision_supported: true
      )
      expect(result).to be_nil
    end

    it "returns nil for nil text blocks" do
      result = described_class.normalize_block(
        { type: "text", text: nil },
        vision_supported: true
      )
      expect(result).to be_nil
    end

    it "passes through unknown block types" do
      result = described_class.normalize_block(
        { type: "custom_type", data: "something" },
        vision_supported: true
      )
      expect(result).to eq({ type: "custom_type", data: "something" })
    end

    it "passes through non-hash blocks" do
      result = described_class.normalize_block("plain string", vision_supported: true)
      expect(result).to eq("plain string")
    end
  end
end
