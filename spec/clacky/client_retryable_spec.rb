# frozen_string_literal: true

require "spec_helper"

RSpec.describe Clacky::Client do
  let(:client) { described_class.new("test-key", base_url: "https://api.example.com", model: "gpt-4") }

  # Helper: build a fake Faraday response
  def fake_response(status:, body:)
    Struct.new(:status, :body).new(status, body)
  end

  # ── check_html_response ───────────────────────────────────────────────────

  describe "#check_html_response (private)" do
    it "raises RetryableError when body starts with <!DOCTYPE" do
      resp = fake_response(status: 200, body: "<!DOCTYPE html><html>500 error</html>")
      expect { client.send(:check_html_response, resp) }
        .to raise_error(Clacky::RetryableError, /HTML error page/)
    end

    it "raises RetryableError when body starts with <!doctype (lowercase)" do
      resp = fake_response(status: 200, body: "<!doctype html><html>bad gateway</html>")
      expect { client.send(:check_html_response, resp) }
        .to raise_error(Clacky::RetryableError, /HTML error page/)
    end

    it "raises RetryableError when body starts with <html" do
      resp = fake_response(status: 200, body: "<html><body>error</body></html>")
      expect { client.send(:check_html_response, resp) }
        .to raise_error(Clacky::RetryableError, /HTML error page/)
    end

    it "does not raise for valid JSON body" do
      resp = fake_response(status: 200, body: '{"content":[]}')
      expect { client.send(:check_html_response, resp) }.not_to raise_error
    end

    it "does not raise for body with leading whitespace before JSON" do
      resp = fake_response(status: 200, body: "  \n{\"content\":[]}")
      expect { client.send(:check_html_response, resp) }.not_to raise_error
    end
  end

  # ── raise_error ───────────────────────────────────────────────────────────

  describe "#raise_error (private)" do
    it "raises RetryableError on 500" do
      resp = fake_response(status: 500, body: '{"error":{"message":"Internal Server Error"}}')
      expect { client.send(:raise_error, resp) }
        .to raise_error(Clacky::RetryableError, /temporarily unavailable/)
    end

    it "raises RetryableError on 502" do
      resp = fake_response(status: 502, body: '{"error":{"message":"Bad Gateway"}}')
      expect { client.send(:raise_error, resp) }
        .to raise_error(Clacky::RetryableError, /temporarily unavailable/)
    end

    it "raises RetryableError on 503" do
      resp = fake_response(status: 503, body: '{"error":{"message":"Service Unavailable"}}')
      expect { client.send(:raise_error, resp) }
        .to raise_error(Clacky::RetryableError, /temporarily unavailable/)
    end

    it "raises RetryableError on 429 rate limit" do
      resp = fake_response(status: 429, body: '{"error":{"message":"Too Many Requests"}}')
      expect { client.send(:raise_error, resp) }
        .to raise_error(Clacky::RetryableError, /Rate limit/)
    end

    it "raises AgentError on 401" do
      resp = fake_response(status: 401, body: '{"error":{"message":"Unauthorized"}}')
      expect { client.send(:raise_error, resp) }
        .to raise_error(Clacky::AgentError, /Invalid API key/)
    end

    it "raises BadRequestError on 400" do
      resp = fake_response(status: 400, body: '{"error":{"message":"Bad Request"}}')
      expect { client.send(:raise_error, resp) }
        .to raise_error(Clacky::BadRequestError, /Bad Request/)
    end

    it "extracts error from chunks when response.body is empty (streaming mode)" do
      # Simulate streaming: response.body is consumed by on_data, chunks hold the error
      resp = fake_response(status: 400, body: "")
      chunks = ['{"error":{"message":"Unknown parameter: stream_options"}}']
      expect { client.send(:raise_error, resp, chunks: chunks) }
        .to raise_error(Clacky::BadRequestError, /Unknown parameter/)
    end
  end

  # ── send_messages_with_tools retry integration ────────────────────────────

  describe "#send_messages_with_tools" do
    let(:messages) { [{ role: "user", content: "hello" }] }

    # Build a fake Faraday connection that always returns the given response
    def stub_openai_connection(client, response)
      req_stub = double("faraday_request")
      opts_stub = double("faraday_options")
      allow(req_stub).to receive(:body=)
      allow(req_stub).to receive(:options).and_return(opts_stub)
      allow(opts_stub).to receive(:timeout=)
      allow(opts_stub).to receive(:on_data=)  # streaming path support

      conn = instance_double(Faraday::Connection)
      allow(conn).to receive(:post).and_yield(req_stub).and_return(response)
      client.instance_variable_set(:@openai_connection, conn)
    end

    context "when server returns HTML with status 200 (gateway error page)" do
      it "falls back to streaming and raises RetryableError if streaming also fails" do
        html_resp = fake_response(status: 200, body: "<!DOCTYPE html><html>502 Bad Gateway</html>")

        # Mock two requests: first non-streaming (HTML) → triggers streaming fallback
        # The streaming fallback also needs a mock (with on_data callback support)
        call_count = 0
        conn = instance_double(Faraday::Connection)
        allow(conn).to receive(:post).twice do |&block|
          req = double("faraday_request")
          opts = double("faraday_options")
          allow(req).to receive(:body=)
          allow(req).to receive(:options).and_return(opts)
          allow(opts).to receive(:timeout=)
          allow(opts).to receive(:on_data=)
          block&.call(req)
          call_count += 1
          call_count == 1 ? html_resp : fake_response(status: 500, body: '{"error":"server error"}')
        end
        client.instance_variable_set(:@openai_connection, conn)

        # The streaming fallback itself encounters a 500 → RetryableError
        expect {
          client.send_messages_with_tools(messages, model: "gpt-4", tools: [], max_tokens: 100)
        }.to raise_error(Clacky::RetryableError)
      end

      it "detects HTML in streaming chunks via check_html_response" do
        client = Clacky::Client.new("sk-test", base_url: "https://test.example.com/v1", model: "gpt-4")

        # Simulate streaming: response.body is empty, chunks contain HTML
        response = double("response", status: 200, body: "")
        html_chunks = ["<!DOCTYPE html><html><body>502 Bad Gateway</body></html>"]

        expect {
          client.send(:check_html_response, response, chunks: html_chunks)
        }.to raise_error(Clacky::RetryableError, /HTML error page/)
      end
    end

    context "when server returns 500" do
      it "raises RetryableError" do
        error_resp = fake_response(status: 500, body: '{"error":{"message":"Internal Server Error"}}')
        stub_openai_connection(client, error_resp)

        expect {
          client.send_messages_with_tools(messages, model: "gpt-4", tools: [], max_tokens: 100)
        }.to raise_error(Clacky::RetryableError, /temporarily unavailable/)
      end
    end
  end
end
