# frozen_string_literal: true

require "spec_helper"
require "digest"
require "json"

RSpec.describe Clacky::Client, "cache affinity" do
  # ── Identity generation ──────────────────────────────────────────────────────

  describe "cache affinity identity generation" do
    it "generates a stable session_id as 32-char hex string (Codex CLI style)" do
      client = described_class.new("test-key", base_url: "https://api.example.com", model: "gpt-4")
      expect(client.cache_affinity_session_id).to match(/\A[0-9a-f]{32}\z/)
    end

    it "generates a stable device_id as 64-char hex string (Claude Code style)" do
      client = described_class.new("test-key", base_url: "https://api.example.com", model: "gpt-4")
      # SHA256("claude_user_#{api_key_hash}") → 64 hex chars
      expect(client.cache_affinity_device_id).to match(/\A[0-9a-f]{64}\z/)
    end

    it "generates a user_id as a JSON string (Claude Code >= 2.1.78 format)" do
      client = described_class.new("test-key", base_url: "https://api.example.com", model: "gpt-4")
      parsed = JSON.parse(client.cache_affinity_user_id)
      expect(parsed).to include("device_id", "account_uuid", "session_id")
      expect(parsed["device_id"]).to match(/\A[0-9a-f]{64}\z/)
      expect(parsed["account_uuid"]).to eq("")
      expect(parsed["session_id"]).to match(/\A[0-9a-f]{32}\z/)
    end

    it "derives user_id from device_id and session_id" do
      client = described_class.new("test-key", base_url: "https://api.example.com", model: "gpt-4")
      parsed = JSON.parse(client.cache_affinity_user_id)
      expect(parsed["device_id"]).to eq(client.cache_affinity_device_id)
      expect(parsed["session_id"]).to eq(client.cache_affinity_session_id)
    end

    it "generates the same device_id for the same API key" do
      client1 = described_class.new("same-key", base_url: "https://api.example.com", model: "gpt-4")
      client2 = described_class.new("same-key", base_url: "https://api.example.com", model: "gpt-4")
      expect(client1.cache_affinity_device_id).to eq(client2.cache_affinity_device_id)
    end

    it "generates different session_ids for different client instances" do
      client1 = described_class.new("test-key", base_url: "https://api.example.com", model: "gpt-4")
      client2 = described_class.new("test-key", base_url: "https://api.example.com", model: "gpt-4")
      expect(client1.cache_affinity_session_id).not_to eq(client2.cache_affinity_session_id)
    end
  end

  # ── inject_cache_affinity! ───────────────────────────────────────────────────

  describe "#inject_cache_affinity! (private)" do
    let(:client) { described_class.new("test-key", base_url: "https://api.example.com", model: "gpt-4") }

    it "injects prompt_cache_key for :openai format (Codex CLI behavior)" do
      body = { model: "gpt-4", messages: [] }
      client.send(:inject_cache_affinity!, body, :openai)
      expect(body[:prompt_cache_key]).to eq(client.cache_affinity_session_id)
      expect(body[:prompt_cache_key]).to match(/\A[0-9a-f]{32}\z/)
    end

    it "injects prompt_cache_key for :responses format (Codex CLI behavior)" do
      body = { model: "gpt-4", input: [] }
      client.send(:inject_cache_affinity!, body, :responses)
      expect(body[:prompt_cache_key]).to eq(client.cache_affinity_session_id)
    end

    it "injects metadata.user_id for :anthropic format (Claude Code >= 2.1.78 JSON format)" do
      body = { model: "claude-3.5-sonnet", messages: [] }
      client.send(:inject_cache_affinity!, body, :anthropic)
      expect(body[:metadata]).to be_a(Hash)
      expect(body[:metadata][:user_id]).to eq(client.cache_affinity_user_id)
      # Verify it's a valid JSON string containing the expected fields
      parsed = JSON.parse(body[:metadata][:user_id])
      expect(parsed).to include("device_id", "account_uuid", "session_id")
    end

    it "preserves existing metadata when injecting user_id" do
      body = { model: "claude-3.5-sonnet", messages: [], metadata: { custom_field: "value" } }
      client.send(:inject_cache_affinity!, body, :anthropic)
      expect(body[:metadata][:custom_field]).to eq("value")
      expect(body[:metadata][:user_id]).to eq(client.cache_affinity_user_id)
    end

    it "does not inject anything for :bedrock format" do
      body = { model: "claude-3.5-sonnet", messages: [] }
      client.send(:inject_cache_affinity!, body, :bedrock)
      expect(body[:prompt_cache_key]).to be_nil
      expect(body[:metadata]).to be_nil
    end

    it "returns the body hash for chaining" do
      body = { model: "gpt-4", messages: [] }
      result = client.send(:inject_cache_affinity!, body, :openai)
      expect(result).to equal(body)  # same object identity
    end
  end
end
