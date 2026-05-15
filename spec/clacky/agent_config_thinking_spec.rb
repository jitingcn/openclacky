# frozen_string_literal: true

require "spec_helper"

RSpec.describe Clacky::AgentConfig do
  describe "per-model thinking config" do
    it "normalizes and persists model-level thinking fields" do
      config = described_class.new(models: [{
        "id" => "1",
        "model" => "gpt-5.4",
        "api_key" => "k",
        "base_url" => "u",
        "type" => "default",
        "thinking_enabled" => "true",
        "reasoning_effort" => "HIGH"
      }])

      expect(config.thinking_enabled).to eq(true)
      expect(config.reasoning_effort).to eq("high")

      yaml = YAML.load(config.to_yaml)
      model = yaml.fetch("models").first
      expect(model["thinking_enabled"]).to eq("true")
      expect(model["reasoning_effort"]).to eq("HIGH")
      expect(yaml.fetch("settings")).not_to have_key("thinking_level")
    end

    it "treats off as disabled for the current model" do
      config = described_class.new(models: [{
        "id" => "1",
        "model" => "gpt-5.4",
        "api_key" => "k",
        "base_url" => "u",
        "type" => "default",
        "thinking_enabled" => "off"
      }])

      expect(config.thinking_enabled).to eq(false)
    end

    it "migrates legacy settings.thinking_level into per-model fields when loading" do
      Dir.mktmpdir("clacky-thinking-migration") do |dir|
        path = File.join(dir, "config.yml")
        File.write(path, YAML.dump(
          "settings" => { "thinking_level" => "medium" },
          "models" => [
            {
              "model" => "gpt-5.4",
              "api_key" => "k",
              "base_url" => "u",
              "type" => "default"
            }
          ]
        ))

        config = described_class.load(path)
        expect(config.thinking_enabled).to eq(true)
        expect(config.reasoning_effort).to eq("medium")
      end
    end

    it "preserves xhigh as an explicit model-level reasoning effort" do
      config = described_class.new(models: [{
        "id" => "1",
        "model" => "gpt-5.4",
        "api_key" => "k",
        "base_url" => "u",
        "type" => "default",
        "reasoning_effort" => "xhigh"
      }])

      expect(config.reasoning_effort).to eq("xhigh")
    end
  end
end
