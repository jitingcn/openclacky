# frozen_string_literal: true

require "spec_helper"

RSpec.describe Clacky::Client, "Kilo Code headers" do
  let(:base_url) { "https://api.example.com" }
  let(:api_key)  { "sk-test-key" }

  def build_client(**overrides)
    described_class.new(api_key, base_url: base_url, model: "gpt-4", **overrides)
  end

  # ── kilo_code_headers helper ──────────────────────────────────────────────

  describe "#kilo_code_headers (private)" do
    it "returns HTTP-Referer: https://kilocode.ai" do
      client = build_client
      headers = client.send(:kilo_code_headers)
      expect(headers["HTTP-Referer"]).to eq("https://kilocode.ai")
    end

    it "returns X-Title: Kilo Code" do
      client = build_client
      headers = client.send(:kilo_code_headers)
      expect(headers["X-Title"]).to eq("Kilo Code")
    end

    it "returns a User-Agent matching Kilo Code format with AI SDK suffix" do
      client = build_client
      headers = client.send(:kilo_code_headers)
      ua = headers["User-Agent"]
      expect(ua).to start_with("Kilo-Code/")
      expect(ua).to include(" ai-sdk/provider-utils/")
      expect(ua).to include(" runtime/node.js/")
    end
  end

  # ── openai_connection headers ─────────────────────────────────────────────

  describe "#openai_connection" do
    it "includes standard auth headers" do
      client = build_client
      conn = client.send(:openai_connection)
      expect(conn.headers["Content-Type"]).to eq("application/json")
      expect(conn.headers["Authorization"]).to eq("Bearer #{api_key}")
    end

    it "includes Kilo Code identity headers" do
      client = build_client
      conn = client.send(:openai_connection)
      expect(conn.headers["HTTP-Referer"]).to eq("https://kilocode.ai")
      expect(conn.headers["X-Title"]).to eq("Kilo Code")
      expect(conn.headers["User-Agent"]).to start_with("Kilo-Code/")
      expect(conn.headers["User-Agent"]).to include(" ai-sdk/provider-utils/")
    end
  end

  # ── responses_connection headers ──────────────────────────────────────────

  describe "#responses_connection" do
    it "includes standard auth headers" do
      client = build_client(api_type: "openai-responses")
      conn = client.send(:responses_connection)
      expect(conn.headers["Content-Type"]).to eq("application/json")
      expect(conn.headers["Authorization"]).to eq("Bearer #{api_key}")
    end

    it "includes Kilo Code identity headers" do
      client = build_client(api_type: "openai-responses")
      conn = client.send(:responses_connection)
      expect(conn.headers["HTTP-Referer"]).to eq("https://kilocode.ai")
      expect(conn.headers["X-Title"]).to eq("Kilo Code")
      expect(conn.headers["User-Agent"]).to start_with("Kilo-Code/")
      expect(conn.headers["User-Agent"]).to include(" ai-sdk/provider-utils/")
    end
  end

  # ── Constants ─────────────────────────────────────────────────────────────

  describe "version constants" do
    it "defines a valid Kilo Code version" do
      version = Clacky::Client::KILO_CODE_VERSION
      expect(version).to match(/\A\d+\.\d+\.\d+\z/)
    end

    it "defines a matching User-Agent string" do
      ua = Clacky::Client::KILO_CODE_UA
      expect(ua).to start_with("Kilo-Code/#{Clacky::Client::KILO_CODE_VERSION}")
      expect(ua).to include(" ai-sdk/provider-utils/")
      expect(ua).to include(" runtime/node.js/")
    end
  end
end
