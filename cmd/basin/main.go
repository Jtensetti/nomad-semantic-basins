package main

import (
	"context"
	"flag"
	"fmt"

	"github.com/Jtensetti/nomad-semantic-basins/basin"
)

func main() {
	text := flag.String("text", "", "text to basinize")
	flag.Parse()
	if *text == "" {
		panic("-text is required")
	}
	e := basin.HashEmbedder{Dims: 512}
	v, err := e.Embed(context.Background(), *text)
	if err != nil {
		panic(err)
	}
	id, err := (basin.Quantizer{}).Basin(v)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%016x\n", id)
}
