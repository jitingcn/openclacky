# frozen_string_literal: true

require "spec_helper"

RSpec.describe Clacky::AgentConfig do
  describe "thinking_level" do
    it "normalizes and persists the configured thinking level" do
      config = described_class.new(models: [{ "id" => "1", "model" => "gpt-5.4", "api_key" => "k", "base_url" => "u", "type" => "default" }], thinking_level: "MEDIUM")

      expect(config.thinking_level).to eq("medium")
      yaml = YAML.load(config.to_yaml)
      expect(yaml.dig("settings", "thinking_level")).to eq("medium")
    end

    it "treats off as disabled" do
      config = described_class.new(models: [{ "id" => "1", "model" => "gpt-5.4", "api_key" => "k", "base_url" => "u", "type" => "default" }], thinking_level: "off")
      expect(config.thinking_level).to be_nil
    end
  end
end
