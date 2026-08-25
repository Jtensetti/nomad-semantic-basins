package basin_test

import (
	"os/exec"
	"strings"
	"testing"
)

// basin is the package that sees a user's query text. Every binary that reads
// a query links it, including the browser core, whose whole architecture rests
// on the statement that the process cannot open a socket.
//
// That statement was false. basin imported net/http directly, for an embedder
// that speaks to a local model over loopback, so importing basin linked a full
// HTTP client and a TLS stack into anything that handled a query --
// github.com/Jtensetti/nomad-browser/selector among them. Nothing constructed
// that embedder anywhere in the tree; the dependency was reached by import
// alone. It now lives in basin/loopback, which a deployment opts into.
//
// This test is why that stays true. It is a dependency-graph check rather than
// a review note, because the difference between "we do not call it" and "it is
// not linked" is the whole point.
func TestTheQueryPackageCannotReachASocket(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "github.com/Jtensetti/nomad-semantic-basins/basin").Output()
	if err != nil {
		t.Fatalf("go list -deps: %v", err)
	}
	linked := map[string]bool{}
	for _, line := range strings.Split(string(out), "\n") {
		linked[strings.TrimSpace(line)] = true
	}
	for _, forbidden := range []string{
		"net",
		"net/http",
		"net/url",
		"crypto/tls",
		"os/exec",
	} {
		if linked[forbidden] {
			t.Errorf("basin links %s, so any binary that handles a query can open a "+
				"socket or launch a process", forbidden)
		}
	}
}

// The companion. The loopback embedder is allowed its dependencies -- that is
// what the split is for -- and this asserts the split actually moved them,
// rather than the package existing while basin kept the import anyway.
func TestTheLoopbackEmbedderIsWhereTheNetworkDependencyLives(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps",
		"github.com/Jtensetti/nomad-semantic-basins/basin/loopback").Output()
	if err != nil {
		t.Fatalf("go list -deps: %v", err)
	}
	if !strings.Contains(string(out), "\nnet/http\n") {
		t.Error("basin/loopback does not link net/http; the embedder it exists to hold " +
			"is not there, and basin may have kept the dependency")
	}
}
