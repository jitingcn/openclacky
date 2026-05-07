# frozen_string_literal: true

RSpec.describe Clacky::AgentConfig, "per-model compression overrides" do
  # Helper to build a config with two models — one with overrides, one without.
  let(:model_with_overrides) do
    {
      "id"                    => "m1",
      "model"                 => "deepseek-v4-pro",
      "api_key"               => "sk-dsk",
      "base_url"              => "https://api.deepseek.com",
      "compression_overrides" => {
        "token_threshold"    => 400_000,
        "message_threshold"  => 300,
        "max_recent_messages" => 30,
        "target_tokens"      => 20_000,
        "idle_threshold"     => 50_000,
        "idle_delay"         => 120
      }
    }
  end

  let(:model_without_overrides) do
    {
      "id"       => "m2",
      "model"    => "gpt-5.3-codex",
      "api_key"  => "sk-gpt",
      "base_url" => "https://api.openai.com"
    }
  end

  let(:config) do
    described_class.new(
      models: [model_with_overrides, model_without_overrides],
      compression_token_threshold: 150_000,
      compression_message_threshold: 200,
      compression_max_recent_messages: 20,
      compression_target_tokens: 10_000,
      idle_compression_threshold: 20_000,
      idle_compression_delay: 180
    )
  end

  describe "effective_* accessors" do
    context "when current model has compression_overrides" do
      before { config.switch_model_by_id("m1") }

      it "returns the model override for token_threshold" do
        expect(config.effective_compression_token_threshold).to eq(400_000)
      end

      it "returns the model override for message_threshold" do
        expect(config.effective_compression_message_threshold).to eq(300)
      end

      it "returns the model override for max_recent_messages" do
        expect(config.effective_compression_max_recent_messages).to eq(30)
      end

      it "returns the model override for target_tokens" do
        expect(config.effective_compression_target_tokens).to eq(20_000)
      end

      it "returns the model override for idle_threshold" do
        expect(config.effective_idle_compression_threshold).to eq(50_000)
      end

      it "returns the model override for idle_delay" do
        expect(config.effective_idle_compression_delay).to eq(120)
      end
    end

    context "when current model has NO compression_overrides" do
      before { config.switch_model_by_id("m2") }

      it "falls back to global default for token_threshold" do
        expect(config.effective_compression_token_threshold).to eq(150_000)
      end

      it "falls back to global default for message_threshold" do
        expect(config.effective_compression_message_threshold).to eq(200)
      end

      it "falls back to global default for max_recent_messages" do
        expect(config.effective_compression_max_recent_messages).to eq(20)
      end

      it "falls back to global default for target_tokens" do
        expect(config.effective_compression_target_tokens).to eq(10_000)
      end

      it "falls back to global default for idle_threshold" do
        expect(config.effective_idle_compression_threshold).to eq(20_000)
      end

      it "falls back to global default for idle_delay" do
        expect(config.effective_idle_compression_delay).to eq(180)
      end
    end

    context "when current model has partial overrides (some keys missing)" do
      let(:partial_model) do
        {
          "id"                    => "m3",
          "model"                 => "partial-model",
          "api_key"               => "sk-partial",
          "base_url"              => "https://api.partial.com",
          "compression_overrides" => {
            "token_threshold" => 300_000
            # only token_threshold overridden; others fall back to global
          }
        }
      end

      let(:config) do
        described_class.new(
          models: [partial_model],
          compression_token_threshold: 150_000,
          compression_message_threshold: 200,
          compression_max_recent_messages: 20,
          compression_target_tokens: 10_000,
          idle_compression_threshold: 20_000,
          idle_compression_delay: 180
        )
      end

      it "returns the override for the overridden key" do
        expect(config.effective_compression_token_threshold).to eq(300_000)
      end

      it "falls back to global for non-overridden keys" do
        expect(config.effective_compression_message_threshold).to eq(200)
        expect(config.effective_compression_max_recent_messages).to eq(20)
        expect(config.effective_compression_target_tokens).to eq(10_000)
        expect(config.effective_idle_compression_threshold).to eq(20_000)
        expect(config.effective_idle_compression_delay).to eq(180)
      end
    end

    context "when switching models" do
      it "immediately reflects the new model's overrides" do
        config.switch_model_by_id("m1")
        expect(config.effective_compression_token_threshold).to eq(400_000)

        config.switch_model_by_id("m2")
        expect(config.effective_compression_token_threshold).to eq(150_000)

        config.switch_model_by_id("m1")
        expect(config.effective_compression_token_threshold).to eq(400_000)
      end
    end
  end

  describe "persistence (to_yaml / load round-trip)" do
    it "persists compression_overrides in the model entry" do
      Dir.mktmpdir do |dir|
        path = File.join(dir, "config.yml")
        config.save(path)

        loaded = described_class.load(path)
        # Find the model with overrides (by name, since id is runtime-only)
        dsk = loaded.models.find { |m| m["model"] == "deepseek-v4-pro" }
        expect(dsk).not_to be_nil
        expect(dsk["compression_overrides"]).to eq({
          "token_threshold"     => 400_000,
          "message_threshold"   => 300,
          "max_recent_messages" => 30,
          "target_tokens"       => 20_000,
          "idle_threshold"      => 50_000,
          "idle_delay"          => 120
        })
      end
    end

    it "does not add compression_overrides key when none are set" do
      Dir.mktmpdir do |dir|
        path = File.join(dir, "config.yml")
        config.save(path)

        loaded = described_class.load(path)
        gpt = loaded.models.find { |m| m["model"] == "gpt-5.3-codex" }
        expect(gpt).not_to be_nil
        expect(gpt).not_to have_key("compression_overrides")
      end
    end
  end

  describe "#add_model with compression_overrides" do
    it "stores compression_overrides on the new model" do
      c = described_class.new(models: [])
      c.add_model(
        model: "test-model",
        api_key: "sk-test",
        base_url: "https://api.test.com",
        compression_overrides: { "token_threshold" => 250_000 }
      )

      expect(c.models.length).to eq(1)
      expect(c.models[0]["compression_overrides"]).to eq({ "token_threshold" => 250_000 })
    end

    it "does not add compression_overrides key when nil" do
      c = described_class.new(models: [])
      c.add_model(model: "test", api_key: "sk", base_url: "https://api.test.com")

      expect(c.models[0]).not_to have_key("compression_overrides")
    end
  end

  describe "PER_MODEL_COMPRESSION_KEYS constant" do
    it "lists all recognized override keys" do
      expected = %w[token_threshold message_threshold max_recent_messages
                     target_tokens idle_threshold idle_delay]
      expect(Clacky::AgentConfig::PER_MODEL_COMPRESSION_KEYS).to eq(expected)
    end
  end
end
