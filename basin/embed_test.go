package basin

import (
	"context"
	"testing"
)

func TestHashEmbedderDeterministic(t *testing.T) {
	e := HashEmbedder{Dims: 256}
	a, err := e.Embed(context.Background(), "Iran military systems")
	if err != nil {
		t.Fatal(err)
	}
	b, err := e.Embed(context.Background(), "Iran military systems")
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != len(b) {
		t.Fatal("length mismatch")
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-deterministic at %d", i)
		}
	}
}

func TestNearbyLexicalInputOftenNearbyBasin(t *testing.T) {
	e := HashEmbedder{Dims: 512}
	a, _ := e.Embed(context.Background(), "Iran military weapons system")
	b, _ := e.Embed(context.Background(), "Iranian military weapon systems")
	q := Quantizer{}
	ba, _ := q.Basin(a)
	bb, _ := q.Basin(b)
	if HammingDistance(ba, bb) > 40 {
		t.Fatalf("unexpectedly distant basins: %d", HammingDistance(ba, bb))
	}
}
