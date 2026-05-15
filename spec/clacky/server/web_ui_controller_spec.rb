# frozen_string_literal: true

require "spec_helper"
require "clacky/server/web_ui_controller"

RSpec.describe Clacky::Server::WebUIController do
  let(:events) { [] }
  let(:broadcaster) do
    lambda do |_session_id, event|
      events << event
    end
  end
  let(:controller) { described_class.new("session-1", broadcaster) }

  describe "#show_tool_call" do
    it "flushes the live assistant stream into its own message before the tool event" do
      controller.show_assistant_delta(reasoning_delta: "Need to inspect the file first.")

      controller.show_tool_call("terminal", { "command" => "ls" })

      expect(events.map { |event| event[:type] }).to eq([
        "assistant_delta",
        "assistant_message",
        "tool_call"
      ])
      expect(events[1]).to include(
        type: "assistant_message",
        content: "",
        reasoning_content: "Need to inspect the file first.",
        files: []
      )
      expect(events[2]).to include(type: "tool_call", name: "terminal")
    end

    it "does not forward the synthesized assistant message to channel subscribers" do
      subscriber = instance_double("ChannelSubscriber")
      controller.subscribe_channel(subscriber)

      expect(subscriber).not_to receive(:show_assistant_message)
      expect(subscriber).to receive(:show_tool_call).with("terminal", { "command" => "pwd" })

      controller.show_assistant_delta(reasoning_delta: "Checking the workspace.")
      controller.show_tool_call("terminal", { "command" => "pwd" })
    end
  end
end
