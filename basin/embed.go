package basin

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"net/http"
	"strings"
	"unicode"
)

type Embedder interface {
	Embed(context.Context, string) ([]float32, error)
}

// HashEmbedder is a deterministic local fallback. It is intentionally simple
// and dependency-free: useful for tests and lexical similarity, not a claim of
// state-of-the-art semantic quality.
type HashEmbedder struct{ Dims int }

func (h HashEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	if h.Dims <= 0 {
		return nil, errors.New("dims must be positive")
	}
	v := make([]float32, h.Dims)
	s := strings.ToLower(strings.TrimSpace(text))
	if s == "" {
		return nil, errors.New("text must not be empty")
	}
	scanner := bufio.NewScanner(strings.NewReader(s))
	scanner.Split(bufio.ScanWords)
	for scanner.Scan() {
		tok := strings.TrimFunc(scanner.Text(), func(r rune) bool { return unicode.IsPunct(r) || unicode.IsSpace(r) })
		if tok == "" {
			continue
		}
		f := fnv.New64a()
		_, _ = f.Write([]byte(tok))
		x := f.Sum64()
		idx := int(x % uint64(h.Dims))
		sign := float32(1)
		if (x>>63)&1 == 1 {
			sign = -1
		}
		v[idx] += sign
		// Character trigrams provide a small amount of morphology robustness.
		rs := []rune(tok)
		for i := 0; i+2 < len(rs); i++ {
			tri := string(rs[i : i+3])
			f.Reset()
			_, _ = f.Write([]byte("3:" + tri))
			y := f.Sum64()
			j := int(y % uint64(h.Dims))
			sg := float32(0.35)
			if (y>>63)&1 == 1 {
				sg = -sg
			}
			v[j] += sg
		}
	}
	normalize(v)
	return v, nil
}

// HTTPEmbedder speaks the common OpenAI-compatible /v1/embeddings shape. It
// can target a local embedding service and does not require a cloud provider.
type HTTPEmbedder struct {
	BaseURL string
	Model   string
	APIKey  string
	Client  *http.Client
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
	if e.BaseURL == "" || e.Model == "" {
		return nil, errors.New("base URL and model are required")
	}
	body, _ := json.Marshal(embeddingRequest{Model: e.Model, Input: text})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(e.BaseURL, "/")+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if e.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.APIKey)
	}
	c := e.Client
	if c == nil {
		c = http.DefaultClient
	}
	resp, err := c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("embedding service returned %s", resp.Status)
	}
	var out embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Data) == 0 || len(out.Data[0].Embedding) == 0 {
		return nil, errors.New("empty embedding response")
	}
	normalize(out.Data[0].Embedding)
	return out.Data[0].Embedding, nil
}

func normalize(v []float32) {
	var ss float64
	for _, x := range v {
		ss += float64(x * x)
	}
	if ss == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(ss))
	for i := range v {
		v[i] *= inv
	}
}

// Quantizer maps normalized vectors to a coarse opaque 64-bit basin using
// deterministic pseudorandom hyperplanes derived from a public epoch seed.
// Basin IDs are routing hints, not cryptographic secrecy primitives.
type Quantizer struct{ Seed [32]byte }

func (q Quantizer) Basin(v []float32) (uint64, error) {
	if len(v) == 0 {
		return 0, errors.New("empty vector")
	}
	var id uint64
	for bit := 0; bit < 64; bit++ {
		var dot float64
		for i, x := range v {
			h := sha256.New()
			_, _ = h.Write(q.Seed[:])
			var buf [16]byte
			binary.BigEndian.PutUint64(buf[:8], uint64(bit))
			binary.BigEndian.PutUint64(buf[8:], uint64(i))
			_, _ = h.Write(buf[:])
			sum := h.Sum(nil)
			u := binary.BigEndian.Uint64(sum[:8])
			// Deterministic coefficient in approximately [-1,1].
			coeff := (float64(u)/float64(^uint64(0)))*2 - 1
			dot += coeff * float64(x)
		}
		if dot >= 0 {
			id |= 1 << uint(bit)
		}
	}
	return id, nil
}

func HammingDistance(a, b uint64) int {
	x := a ^ b
	n := 0
	for x != 0 {
		x &= x - 1
		n++
	}
	return n
}
