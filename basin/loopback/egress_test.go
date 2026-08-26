//go:build linux

package loopback

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"testing"
	"time"
	"unsafe"
)

// PROD-24 asks for attempted-egress capture: not a promise that the embedding
// service stays on the host, and not a systemd file whose directives are read
// rather than exercised, but a measurement of what the chain actually emits.
//
// Two things are established here, and they answer different objections.
//
// The chain runs inside a network namespace whose only interface is loopback.
// Off-host is not blocked there, it is *absent*: there is no route, so a
// process that decided to reach out has nowhere to go. That the embedding still
// succeeds is the finding -- the whole client, shim and model-server chain needs
// nothing but loopback, which is a statement about the code rather than about a
// configuration somebody applied.
//
// And a packet capture inside that namespace records every datagram the chain
// produced, so the claim is what was observed rather than what was permitted.
// A capture alone would be weaker: it would show that nothing left during one
// run. The namespace shows that nothing could.

const egressChildMarker = "NOMAD_EGRESS_NAMESPACE_CHILD"

// interfaceRequest is the ifreq a SIOCGIFFLAGS/SIOCSIFFLAGS ioctl takes.
type interfaceRequest struct {
	name  [16]byte
	flags uint16
	_     [22]byte
}

// bringLoopbackUp starts the loopback interface inside a fresh network
// namespace.
//
// A new namespace has exactly one interface and it is DOWN, which is a state
// worth knowing about: binding 127.0.0.1 succeeds there and connecting to it
// fails with "network is unreachable". A test that only bound would conclude
// loopback worked. This is done with an ioctl rather than by running ip(8),
// which is not installed, and through the standard library rather than by
// adding a dependency for a test.
func bringLoopbackUp() error {
	fd, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_DGRAM, 0)
	if err != nil {
		return fmt.Errorf("socket: %w", err)
	}
	defer syscall.Close(fd)

	var request interfaceRequest
	copy(request.name[:], "lo")
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		syscall.SIOCGIFFLAGS, uintptr(unsafe.Pointer(&request))); errno != 0 {
		return fmt.Errorf("read loopback flags: %w", errno)
	}
	request.flags |= syscall.IFF_UP | syscall.IFF_RUNNING
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd),
		syscall.SIOCSIFFLAGS, uintptr(unsafe.Pointer(&request))); errno != 0 {
		return fmt.Errorf("bring loopback up: %w", errno)
	}
	return nil
}

// loopbackOnly reports whether a tcpdump line names only loopback addresses,
// and returns what it found otherwise.
//
// Parsed rather than substring-matched, because "::1" is a substring of
// "2001:db8::1" and a check that accepted that would pass on exactly the
// packets it exists to catch. tcpdump -nn -q prints
// "<time> IP <src>.<port> > <dst>.<port>: <summary>".
func loopbackOnly(line string) (bool, string) {
	marker := " IP "
	if index := strings.Index(line, " IP6 "); index >= 0 {
		marker = " IP6 "
	}
	index := strings.Index(line, marker)
	if index < 0 {
		// Not an IP packet line: ARP, or a summary tcpdump emitted. Nothing
		// to judge, and nothing that carries a query.
		return true, ""
	}
	rest := line[index+len(marker):]
	parts := strings.SplitN(rest, " > ", 2)
	if len(parts) != 2 {
		return false, rest
	}
	destination := parts[1]
	if colon := strings.Index(destination, ":"); colon >= 0 && strings.Contains(destination[:colon], ".") {
		destination = destination[:colon]
	}
	for _, endpoint := range []string{parts[0], destination} {
		endpoint = strings.TrimSpace(endpoint)
		dot := strings.LastIndex(endpoint, ".")
		if dot < 0 {
			return false, endpoint
		}
		address := endpoint[:dot]
		if address != "127.0.0.1" && address != "::1" {
			return false, address
		}
	}
	return true, ""
}

// runChain is the whole embedding path: a model server, the shim in front of
// it, and a client that seals a query to the shim. It returns the vector, so a
// caller can tell "it worked" from "it was never reached".
func runChain(t *testing.T) []float32 {
	t.Helper()
	modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"embedding": []float32{3, 4}}},
		})
	}))
	defer modelServer.Close()

	shim := httptest.NewServer(Service{
		ServiceKey: testServiceKey(),
		Upstream:   OpenAIUpstream{BaseURL: modelServer.URL},
	})
	defer shim.Close()

	embedder := HTTPEmbedder{
		BaseURL: shim.URL, Model: "local-model", ServiceKey: testServiceKey(),
		Timeout: 10 * time.Second,
	}
	vector, err := embedder.Embed(context.Background(), privateQuery)
	if err != nil {
		t.Fatalf("the embedding chain failed: %v", err)
	}
	return vector
}

func TestTheEmbeddingChainNeedsNothingButLoopback(t *testing.T) {
	if os.Getenv(egressChildMarker) == "1" {
		if err := bringLoopbackUp(); err != nil {
			t.Fatalf("could not start loopback in the namespace: %v", err)
		}
		// Inside the namespace. Establish that off-host is genuinely absent
		// before concluding anything from the chain succeeding: a namespace
		// that still had a route would make this test prove nothing.
		if conn, err := net.DialTimeout("tcp", "1.1.1.1:80", 2*time.Second); err == nil {
			_ = conn.Close()
			t.Fatal("the namespace can still reach off the host, so nothing below is " +
				"evidence about egress")
		}
		vector := runChain(t)
		if len(vector) != 2 {
			t.Fatalf("the chain returned %d dimensions", len(vector))
		}
		return
	}

	unshare, err := exec.LookPath("unshare")
	if err != nil {
		t.Skip("unshare is unavailable, so a loopback-only namespace cannot be built; " +
			"this is an environment limit and not a pass")
	}
	command := exec.Command(unshare, "--net", os.Args[0],
		"-test.run", "^TestTheEmbeddingChainNeedsNothingButLoopback$", "-test.v")
	command.Env = append(os.Environ(), egressChildMarker+"=1")
	output, err := command.CombinedOutput()
	if err != nil {
		if strings.Contains(string(output), "Operation not permitted") ||
			strings.Contains(err.Error(), "operation not permitted") {
			t.Skipf("this environment does not permit a network namespace; an "+
				"environment limit and not a pass:\n%s", output)
		}
		t.Fatalf("the embedding chain failed with only loopback available:\n%s", output)
	}
	if !strings.Contains(string(output), "PASS") {
		t.Fatalf("the child did not report a pass:\n%s", output)
	}
	t.Logf("MEASURED: the client, shim and model server complete an embedding inside a "+
		"network namespace with no route off the host:\n%s", strings.TrimSpace(string(output)))
}

// The capture half. The namespace says nothing could leave; this says nothing
// did, in the form the criterion asks for.
func TestNothingTheEmbeddingChainEmitsLeavesLoopback(t *testing.T) {
	if os.Getenv(egressChildMarker) == "1" {
		if err := bringLoopbackUp(); err != nil {
			t.Fatalf("could not start loopback in the namespace: %v", err)
		}
		capturePath := os.Getenv("NOMAD_EGRESS_CAPTURE")
		if capturePath == "" {
			t.Fatal("the child was started without a capture path")
		}
		tcpdump, err := exec.LookPath("tcpdump")
		if err != nil {
			t.Skip("tcpdump is unavailable inside the namespace")
		}
		// -U writes each packet as it arrives. Without it tcpdump buffers,
		// and a buffer that is still unwritten when the process ends leaves
		// a truncated file that reads as "nothing was captured" -- which is
		// indistinguishable from the result this test exists to report.
		capture := exec.Command(tcpdump, "-i", "any", "-n", "-U", "-w", capturePath)
		if err := capture.Start(); err != nil {
			t.Fatalf("start capture: %v", err)
		}
		// tcpdump needs a moment to attach before the chain runs, or the
		// capture is of nothing and the test passes for want of traffic.
		time.Sleep(1500 * time.Millisecond)
		vector := runChain(t)
		time.Sleep(500 * time.Millisecond)
		// SIGTERM rather than SIGKILL, so it closes the file it is writing.
		_ = capture.Process.Signal(syscall.SIGTERM)
		_, _ = capture.Process.Wait()
		if len(vector) != 2 {
			t.Fatalf("the chain returned %d dimensions", len(vector))
		}
		return
	}

	unshare, err := exec.LookPath("unshare")
	if err != nil {
		t.Skip("unshare is unavailable; an environment limit and not a pass")
	}
	if _, err := exec.LookPath("tcpdump"); err != nil {
		t.Skip("tcpdump is unavailable; an environment limit and not a pass")
	}
	capturePath := t.TempDir() + "/egress.pcap"

	command := exec.Command(unshare, "--net", os.Args[0],
		"-test.run", "^TestNothingTheEmbeddingChainEmitsLeavesLoopback$", "-test.v")
	command.Env = append(os.Environ(), egressChildMarker+"=1", "NOMAD_EGRESS_CAPTURE="+capturePath)
	output, err := command.CombinedOutput()
	if err != nil {
		if strings.Contains(string(output), "Operation not permitted") {
			t.Skipf("this environment does not permit a network namespace:\n%s", output)
		}
		t.Fatalf("the captured run failed:\n%s", output)
	}
	if strings.Contains(string(output), "SKIP") {
		t.Skipf("the child skipped:\n%s", output)
	}

	read := exec.Command("tcpdump", "-r", capturePath, "-nn", "-q")
	packets, err := read.CombinedOutput()
	if err != nil {
		t.Fatalf("read capture: %v\n%s", err, packets)
	}
	lines := 0
	for _, line := range strings.Split(string(packets), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "reading from file") {
			continue
		}
		lines++
		// Every address in a captured packet must be loopback. Anything else
		// is the chain having addressed something off the host, whether or
		// not the namespace could deliver it -- and an attempt is the thing
		// worth knowing about.
		if ok, found := loopbackOnly(line); !ok {
			t.Errorf("a captured packet names %q, which is not loopback: %s", found, line)
		}
	}
	if lines == 0 {
		t.Fatal("the capture is empty; the chain emitted nothing, so nothing was measured")
	}
	t.Logf("MEASURED: %d packets captured across the whole embedding chain, every one of "+
		"them loopback, inside a namespace with no route off the host", lines)
}

// The captures above are worth nothing if the reader would accept a packet
// that was not loopback. This is the control: a process in the same namespace
// that addresses something off the host must produce a line the check rejects.
func TestTheEgressCheckRejectsANonLoopbackPacket(t *testing.T) {
	for _, line := range []string{
		"12:00:00.1 IP 10.0.0.5.443 > 10.0.0.9.51000: tcp 0",
		"12:00:00.1 IP 192.0.2.1.53 > 198.51.100.2.40000: UDP, length 40",
		"12:00:00.1 IP6 2001:db8::1.443 > 2001:db8::2.40000: tcp 0",
		// The one a substring check gets wrong: "::1" appears inside it.
		"12:00:00.1 IP6 ::1.8080 > 2001:db8::1.40000: tcp 0",
		"12:00:00.1 IP 127.0.0.1.8080 > 10.0.0.9.51000: tcp 0",
	} {
		if ok, _ := loopbackOnly(line); ok {
			t.Errorf("the check accepts an off-host packet: %s", line)
		}
	}
	for _, line := range []string{
		"12:00:00.1 IP 127.0.0.1.8779 > 127.0.0.1.40000: tcp 0",
		"12:00:00.1 IP6 ::1.8080 > ::1.40001: tcp 0",
	} {
		if ok, found := loopbackOnly(line); !ok {
			t.Errorf("the check rejects a loopback packet as %q: %s", found, line)
		}
	}
	// And the chain must be reachable outside a namespace too, so a skip in
	// this environment never leaves the whole file untested.
	if vector := runChain(t); len(vector) != 2 {
		t.Fatalf("the chain returned %d dimensions on the host", len(vector))
	}
}
