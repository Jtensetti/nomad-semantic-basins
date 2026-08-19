# nomad-semantic-basins

Working research implementation of Nomad's **semantic basin** layer.

It separates human meaning from data transport. A local embedder converts text to a vector; a coarse random-hyperplane quantizer converts that vector to an opaque basin ID suitable for probabilistic routing experiments.

## Implemented

- `Embedder` interface.
- Dependency-free deterministic `HashEmbedder` for tests and lexical fallback.
- `HTTPEmbedder` for any local OpenAI-compatible embedding endpoint.
- Epoch-seedable 64-bit random-hyperplane basin quantizer.
- Hamming-distance utility and tests.

## Important limitation

A basin ID is **not** a privacy guarantee. Embeddings and coarse semantic buckets can leak meaning. Production privacy would require a reviewed private-retrieval/aggregation design. This repository intentionally does not pretend otherwise.

## Build

```bash
go test ./...
go test -race ./...
go vet ./...
go run ./cmd/basin -text 'vapensystem i irans militär'
```
