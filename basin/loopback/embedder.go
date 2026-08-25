// Package loopback holds the embedder that speaks to a local embedding
// service over HTTP.
//
// It is a separate package for one reason: importing it links net, net/http
// and crypto/tls, and the package that handles a user's query text must not
// drag a TLS stack and a full HTTP client into every binary that reads a
// query. The browser core's architecture rests on the statement that the
// process cannot open a socket, and a socket-capable transitive dependency
// inside basin made that statement false for anything importing basin --
// including github.com/Jtensetti/nomad-browser/selector, which is the private
// selection side.
//
// Nothing in the Nomad tree constructed this embedder; it is an integration
// point for a deployment that runs a local model. Keeping it here means a
// deployment opts into the dependency explicitly, and that basin itself is
// socket-free by construction rather than by review.
package loopback

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/Jtensetti/nomad-semantic-basins/basin"
)

// HTTPEmbedder speaks the common OpenAI-compatible /v1/embeddings
// request shape. It accepts only literal loopback IPs, disables proxy use and
// rejects redirects so private query text cannot leave the host through normal
// HTTP client configuration.
type HTTPEmbedder struct {
	BaseURL          string
	Model            string
	APIKey           string
	Timeout          time.Duration
	MaxInputBytes    int
	MaxDimensions    int
	MaxResponseBytes int
}

type embeddingRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type embeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

func (e HTTPEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil, errors.New("text must not be empty")
	}
	maxInput, err := basin.BoundedInt(e.MaxInputBytes, basin.DefaultMaxInputBytes, basin.HardMaxInputBytes, "maximum input size")
	if err != nil {
		return nil, err
	}
	if len(trimmed) > maxInput {
		return nil, errors.New("text exceeds maximum input size")
	}
	maxDimensions, err := basin.BoundedInt(e.MaxDimensions, basin.DefaultMaxEmbeddingDims, basin.HardMaxEmbeddingDims, "maximum dimensions")
	if err != nil {
		return nil, err
	}
	maxResponse, err := basin.BoundedInt(e.MaxResponseBytes, basin.DefaultMaxEmbeddingResponse, basin.HardMaxEmbeddingResponse, "maximum response size")
	if err != nil {
		return nil, err
	}
	if e.BaseURL == "" || e.Model == "" {
		return nil, errors.New("base URL and model are required")
	}
	base, err := url.Parse(e.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("parse base URL: %w", err)
	}
	if base.Scheme != "http" {
		return nil, errors.New("loopback embedding endpoint must use http")
	}
	if base.User != nil || base.RawQuery != "" || base.Fragment != "" {
		return nil, errors.New("embedding base URL must not contain user info, query or fragment")
	}
	ip := net.ParseIP(base.Hostname())
	if ip == nil || !ip.IsLoopback() {
		return nil, errors.New("embedding endpoint host must be a literal loopback IP")
	}

	body, err := json.Marshal(embeddingRequest{Model: e.Model, Input: trimmed})
	if err != nil {
		return nil, err
	}
	endpoint := strings.TrimRight(e.BaseURL, "/") + "/v1/embeddings"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if e.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.APIKey)
	}

	timeout := e.Timeout
	if timeout <= 0 {
		timeout = 10 * time.Second
	}
	client := &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			Proxy:             nil,
			DisableKeepAlives: true,
		},
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return errors.New("embedding endpoint redirects are disabled")
		},
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("embedding service returned %s", resp.Status)
	}
	payload, err := io.ReadAll(io.LimitReader(resp.Body, int64(maxResponse)+1))
	if err != nil {
		return nil, err
	}
	if len(payload) > maxResponse {
		return nil, errors.New("embedding response exceeds maximum size")
	}
	var out embeddingResponse
	if err := json.Unmarshal(payload, &out); err != nil {
		return nil, err
	}
	if len(out.Data) != 1 || len(out.Data[0].Embedding) == 0 {
		return nil, errors.New("embedding response must contain exactly one non-empty vector")
	}
	if len(out.Data[0].Embedding) > maxDimensions {
		return nil, errors.New("embedding response exceeds maximum dimensions")
	}
	if !basin.Normalize(out.Data[0].Embedding) {
		return nil, errors.New("embedding response was the zero vector")
	}
	return out.Data[0].Embedding, nil
}
