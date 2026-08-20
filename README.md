# nomad-semantic-basins

Local vector-to-basin experiments for Nomad discovery.

The repository separates **embedding** from **quantization**. A caller supplies a vector representation; `Quantizer` maps it to a 64-bit random-hyperplane signature. Hamming distance between signatures is a lossy similarity hint, not an identity or confidentiality primitive.

## Embedders

- `LexicalHashEmbedder` is a deterministic word/character-ngram hashing baseline. It is lexical, not semantic, and exists mainly for tests and offline development.
- `LoopbackHTTPEmbedder` speaks an OpenAI-compatible local embeddings request shape. It accepts only literal loopback IPs over HTTP, disables proxies and rejects redirects. The implementation intentionally does not accept a caller-supplied HTTP client because that would reopen proxy/redirect escape paths for private query text.
- Both embedders bound private input length and vector dimensions. The loopback adapter also bounds the complete JSON response before decoding it, rejects multiple vectors and disables HTTP keep-alives.

## Quantizer

`Quantizer` derives deterministic standard-normal hyperplanes from a public seed and takes the sign of each projection (SimHash-style random hyperplane LSH). Tests check exact determinism, the expected complement for opposite vectors, and that angularly close vectors are materially closer than orthogonal vectors across many deterministic seeds.

## Privacy caveat

Basin identifiers are metadata, not secrets. Similar inputs are intentionally more likely to have similar signatures. Any network design that exposes basin IDs needs a separate inversion, membership-inference and aggregation-leakage analysis.

```bash
go test -race ./...
go vet ./...
go run ./cmd/basin -text 'example text'
```
