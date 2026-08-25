package loopback

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// The half of TestEmbeddersRejectOversizedInputsAndDimensions that belongs to
// this embedder. The bound comes from basin.BoundedInt, so the point is that
// an embedder in another package reaches the same limit rather than its own.
func TestHTTPEmbedderRejectsOversizedInput(t *testing.T) {
	embedder := HTTPEmbedder{
		BaseURL: "http://127.0.0.1:9", Model: "local", MaxInputBytes: 4,
	}
	if _, err := embedder.Embed(context.Background(), "private query"); err == nil {
		t.Fatal("loopback embedder accepted oversized input")
	}
}

// The endpoint check is what stops private query text leaving the host. It is
// the reason this package can exist at all, so it is checked against more than
// one way of not being loopback.
func TestHTTPEmbedderAcceptsOnlyALiteralLoopbackAddress(t *testing.T) {
	for _, base := range []string{
		"http://example.com",
		"http://localhost:9",
		"http://0.0.0.0:9",
		"http://169.254.169.254",
		"http://10.0.0.1:9",
		"https://127.0.0.1:9",
		"http://user:pass@127.0.0.1:9",
		"http://127.0.0.1:9?to=elsewhere",
	} {
		embedder := HTTPEmbedder{BaseURL: base, Model: "test"}
		if _, err := embedder.Embed(context.Background(), "private query"); err == nil {
			t.Errorf("%s was accepted as a loopback embedding endpoint", base)
		}
	}
	// And the addresses that are loopback are not rejected for being so: this
	// one fails to connect, which is a different error from being refused.
	for _, base := range []string{"http://127.0.0.1:9", "http://[::1]:9"} {
		embedder := HTTPEmbedder{BaseURL: base, Model: "test", Timeout: 250 * time.Millisecond}
		_, err := embedder.Embed(context.Background(), "private query")
		if err == nil {
			continue
		}
		if strings.Contains(err.Error(), "literal loopback IP") {
			t.Errorf("%s was refused as non-loopback", base)
		}
	}
}

func TestHTTPEmbedderRejectsRemoteHost(t *testing.T) {
	e := HTTPEmbedder{BaseURL: "http://example.com", Model: "test"}
	if _, err := e.Embed(context.Background(), "private query"); err == nil {
		t.Fatal("expected non-loopback endpoint to be rejected")
	}
}

func TestHTTPEmbedderRejectsRedirect(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "http://127.0.0.1:9/", http.StatusTemporaryRedirect)
	}))
	defer server.Close()

	e := HTTPEmbedder{BaseURL: server.URL, Model: "local-model"}
	if _, err := e.Embed(context.Background(), "private query"); err == nil {
		t.Fatal("expected redirect to be rejected")
	}
}

func TestHTTPEmbedderRequestAndNormalization(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		if req["model"] != "local-model" || req["input"] != "private query" {
			t.Fatalf("unexpected request: %#v", req)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"embedding": []float32{3, 4}}},
		})
	}))
	defer server.Close()

	e := HTTPEmbedder{BaseURL: server.URL, Model: "local-model"}
	v, err := e.Embed(context.Background(), "private query")
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != 2 || math.Abs(float64(v[0]-.6)) > 1e-6 || math.Abs(float64(v[1]-.8)) > 1e-6 {
		t.Fatalf("unexpected normalized vector: %#v", v)
	}
}

func TestHTTPEmbedderBoundsResponseAndDimensions(t *testing.T) {
	t.Run("response bytes", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write(bytes.Repeat([]byte{'x'}, 1024))
		}))
		defer server.Close()
		e := HTTPEmbedder{
			BaseURL: server.URL, Model: "local", MaxResponseBytes: 64,
		}
		if _, err := e.Embed(context.Background(), "query"); err == nil {
			t.Fatal("accepted oversized embedding response")
		}
	})

	t.Run("vector dimensions", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"embedding": []float32{1, 2, 3}}},
			})
		}))
		defer server.Close()
		e := HTTPEmbedder{
			BaseURL: server.URL, Model: "local", MaxDimensions: 2,
		}
		if _, err := e.Embed(context.Background(), "query"); err == nil {
			t.Fatal("accepted excessive embedding dimensions")
		}
	})

	t.Run("multiple vectors", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{
					{"embedding": []float32{1, 2}},
					{"embedding": []float32{3, 4}},
				},
			})
		}))
		defer server.Close()
		e := HTTPEmbedder{BaseURL: server.URL, Model: "local"}
		if _, err := e.Embed(context.Background(), "query"); err == nil {
			t.Fatal("accepted multiple embedding vectors")
		}
	})
}
