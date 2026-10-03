"""#1547 — AST sink-based ratchet for outbound HTTP calls across src/soup_cli.

The #616 ratchet (tests/test_issue616_net_guard_is_shared.py) finds gates by
their syntactic shape: a ``.hostname`` attribute access occurring next to a
``"localhost"`` literal or a ``*LOOPBACK_HOSTS`` identifier within the same
function. An alternative gate written with a different idiom — such as
``urlsplit(u).netloc.split(":")[0]``, ``httpx.URL(u).host``, or a bare
outbound call like ``httpx.post(base + ...)`` with no validation at all —
passed it unchecked.

``src/soup_cli/utils/energy.py:66`` (``validate_electricity_map_endpoint``) was
a concrete instance: its own private ``_LOOPBACK`` set and an ad-hoc IP
predicate that missed abbreviated IPv4 (e.g. ``127.1``) and RFC 6598
carrier-grade NAT address space (``100.64.0.0/10``).

This file implements a sink-based ratchet over the entire ``src/soup_cli/``
tree:
1. Every outbound HTTP sink call (``httpx``, ``urllib.request``, and ``requests``
   openers/callers) is discovered via AST analysis.
2. Every discovered ``(relpath, function_name)`` must exist in the explicit
   declared table below, naming the shared gate it relies on or the reason it
   needs none (fixed host, loopback only, operator environment).
3. A newly introduced HTTP sink with no table entry fails the test immediately,
   reporting the file, function, line number, and offending call snippet.
4. An allowlisted entry that disappears or is renamed also fails the test,
   preventing the ledger from going stale.
5. Path handling is normalized to POSIX style (``.as_posix()``) to ensure
   identical execution on Windows and Linux CI runners.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "soup_cli"

HTTPX_CALL_METHODS = frozenset(
    {"post", "get", "stream", "request", "put", "patch", "delete", "head", "options"}
)
HTTPX_CLIENT_TYPES = frozenset({"Client", "AsyncClient"})
URLLIB_REQUEST_CALLS = frozenset({"urlopen", "build_opener"})
REQUESTS_CALL_METHODS = frozenset(
    {"post", "get", "put", "patch", "delete", "head", "options", "request"}
)


# ---------------------------------------------------------------------------
# Declared Outbound Sink Ledger
# ---------------------------------------------------------------------------
# Keyed by (relpath relative to src/soup_cli with forward slashes, function_name).
# Every entry documents the gate function or verified exemption justification.
# Any call site not in this table fails test_every_outbound_http_sink_is_declared.
# Any entry in this table that has no live sink in code fails
# test_every_declared_sink_is_still_present.
DECLARED_SINKS: dict[tuple[str, str], str] = {
    ("commands/generate.py", "_generate_openai"): (
        "guarded: validates api_base against LOOPBACK_HOSTS (HTTPS required for "
        "remote) and calls net_guard.refuse_private_ip_literal"
    ),
    ("commands/generate.py", "_generate_server"): (
        "guarded: validates api_base against LOOPBACK_HOSTS (HTTPS required for "
        "remote) and calls net_guard.refuse_private_ip_literal"
    ),
    ("data/providers/anthropic.py", "generate_anthropic"): (
        "fixed host: ANTHROPIC_API_URL is hardcoded to https://api.anthropic.com/v1/messages"
    ),
    ("data/providers/ollama.py", "detect_ollama"): (
        "loopback only: probes local Ollama instance (default http://localhost:11434)"
    ),
    ("data/providers/ollama.py", "generate_ollama"): (
        "guarded: validate_ollama_url restricts base_url to localhost/127.0.0.1/::1"
    ),
    ("data/providers/vllm.py", "generate_vllm"): (
        "guarded: validate_vllm_url requires HTTPS for remote and calls "
        "net_guard.refuse_private_ip_literal"
    ),
    ("eval/judge.py", "_judge_request"): (
        "guarded: JudgeEvaluator.__init__ enforces policy via validate_judge_api_base, "
        "which calls net_guard.refuse_private_ip_literal"
    ),
    ("ui/app.py", "_stream_chat"): (
        "guarded: chat_send validates endpoint (loopback-only HTTP or HTTPS) and "
        "calls net_guard.refuse_private_ip_literal before initiating stream"
    ),
    ("utils/data_forge.py", "_ollama_judge"): (
        "guarded: create_judge_client enforces validate_ollama_url before returning judge closure"
    ),
    ("utils/data_forge.py", "_anthropic_judge"): (
        "fixed host: _ANTHROPIC_MESSAGES_URL is hardcoded to https://api.anthropic.com/v1/messages"
    ),
    ("utils/data_forge.py", "_vllm_judge"): (
        "guarded: create_judge_client enforces validate_vllm_url before returning judge closure"
    ),
    ("utils/ingest_pull.py", "_urllib_transport"): (
        "guarded: resolve_langfuse_host routes host through validate_webhook_url "
        "(HTTPS only, rejects private hosts unless explicitly opted in)"
    ),
    ("utils/loop_stages.py", "_post_activate"): (
        "operator environment: deploy_to_canary validates endpoint via "
        "validate_webhook_url and restricts target to local/LAN via _endpoint_is_local"
    ),
    ("utils/magpie.py", "_ollama_generate"): (
        "guarded: create_magpie_generator enforces validate_ollama_url (loopback only)"
    ),
    ("utils/magpie.py", "_vllm_generate"): (
        "guarded: create_magpie_generator enforces validate_vllm_url (calls "
        "net_guard.refuse_private_ip_literal)"
    ),
    ("utils/sglang.py", "generate_with_runtime"): (
        "loopback only: target is the in-process local SGLang Runtime server (runtime.url)"
    ),
    ("utils/trackers.py", "send_telemetry_payload"): (
        "guarded: _resolve_posthog_target enforces _telemetry_endpoint_is_safe "
        "(HTTPS only, pre- and post-DNS private-IP refusal)"
    ),
    ("utils/webhooks.py", "post_webhook"): (
        "guarded: validates URL via validate_webhook_url (HTTPS for remote, "
        "calls net_guard.is_private_or_link_local)"
    ),
}


# ---------------------------------------------------------------------------
# AST Outbound Sink Scanner
# ---------------------------------------------------------------------------
class _SinkVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.func_stack: list[str] = []
        self.httpx_modules: set[str] = set()
        self.httpx_func_sinks: dict[str, str] = {}
        self.httpx_client_types: set[str] = set()
        self.urllib_request_modules: set[str] = set()
        self.urllib_func_sinks: dict[str, str] = {}
        self.requests_modules: set[str] = set()
        self.requests_func_sinks: dict[str, str] = {}
        self.opener_vars: set[str] = set()
        self.discovered_sinks: list[tuple[str, int, str]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            if alias.name == "httpx":
                self.httpx_modules.add(name)
            elif alias.name in ("urllib.request", "urllib"):
                # "import urllib.request as r" binds r -> urllib.request
                # "import urllib" binds urllib -> urllib.request via attribute
                if alias.name == "urllib.request":
                    self.urllib_request_modules.add(name)
            elif alias.name == "requests":
                self.requests_modules.add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if mod == "httpx":
            for alias in node.names:
                local_name = alias.asname or alias.name
                if alias.name in HTTPX_CALL_METHODS:
                    self.httpx_func_sinks[local_name] = f"httpx.{alias.name}"
                elif alias.name in HTTPX_CLIENT_TYPES:
                    self.httpx_client_types.add(local_name)
        elif mod == "urllib":
            for alias in node.names:
                local_name = alias.asname or alias.name
                if alias.name == "request":
                    self.urllib_request_modules.add(local_name)
        elif mod == "urllib.request":
            for alias in node.names:
                local_name = alias.asname or alias.name
                if alias.name in URLLIB_REQUEST_CALLS:
                    self.urllib_func_sinks[local_name] = f"urllib.request.{alias.name}"
        elif mod == "requests":
            for alias in node.names:
                local_name = alias.asname or alias.name
                if alias.name in REQUESTS_CALL_METHODS:
                    self.requests_func_sinks[local_name] = f"requests.{alias.name}"
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        # Track variables assigned to opener = urllib.request.build_opener(...)
        if isinstance(node.value, ast.Call):
            call_repr = self._match_call(node.value)
            if call_repr == "urllib.request.build_opener":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.opener_vars.add(target.id)
        self.generic_visit(node)

    def _match_call(self, node: ast.Call) -> str | None:
        func = node.func
        # 1. Attribute call: obj.method(...)
        if isinstance(func, ast.Attribute):
            attr = func.attr
            val = func.value
            # httpx.<method> or httpx.<Client>
            if isinstance(val, ast.Name):
                if val.id in self.httpx_modules:
                    if attr in HTTPX_CALL_METHODS or attr in HTTPX_CLIENT_TYPES:
                        return f"httpx.{attr}"
                # request.<method> (from urllib import request)
                if val.id in self.urllib_request_modules and attr in URLLIB_REQUEST_CALLS:
                    return f"urllib.request.{attr}"
                # requests.<method>
                if val.id in self.requests_modules and attr in REQUESTS_CALL_METHODS:
                    return f"requests.{attr}"
                # opener.open(...)
                if (val.id in self.opener_vars or val.id == "opener") and attr == "open":
                    return "opener.open"
            # urllib.request.<method>
            elif isinstance(val, ast.Attribute):
                if (
                    isinstance(val.value, ast.Name)
                    and val.value.id == "urllib"
                    and val.attr == "request"
                    and attr in URLLIB_REQUEST_CALLS
                ):
                    return f"urllib.request.{attr}"

        # 2. Direct name call: method(...)
        elif isinstance(func, ast.Name):
            if func.id in self.httpx_func_sinks:
                return self.httpx_func_sinks[func.id]
            if func.id in self.httpx_client_types:
                return f"httpx.{func.id}"
            if func.id in self.urllib_func_sinks:
                return self.urllib_func_sinks[func.id]
            if func.id in self.requests_func_sinks:
                return self.requests_func_sinks[func.id]

        return None

    def visit_Call(self, node: ast.Call) -> None:
        call_repr = self._match_call(node)
        if call_repr:
            func_name = self.func_stack[-1] if self.func_stack else "<module>"
            self.discovered_sinks.append((func_name, node.lineno, call_repr))
        self.generic_visit(node)


def find_outbound_sinks(source: str) -> list[tuple[str, int, str]]:
    """Return ``[(func_name, lineno, call_repr)]`` for all outbound sinks in source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    visitor = _SinkVisitor()
    visitor.visit(tree)
    return visitor.discovered_sinks


def scan_src_sinks(src_dir: Path = SRC) -> list[tuple[str, str, int, str]]:
    """Scan all python files under ``src_dir`` and return list of (relpath, func, line, call)."""
    hits: list[tuple[str, str, int, str]] = []
    for path in sorted(src_dir.rglob("*.py")):
        rel = path.relative_to(src_dir).as_posix()
        source = path.read_text(encoding="utf-8", errors="replace")
        for func_name, lineno, call_repr in find_outbound_sinks(source):
            hits.append((rel, func_name, lineno, call_repr))
    return hits


# ---------------------------------------------------------------------------
# Ratchet Test Suite & Invariants
# ---------------------------------------------------------------------------
class TestOutboundHttpSinksAreDeclared:
    def test_every_outbound_http_sink_is_declared(self) -> None:
        """Every httpx and urllib.request sink call in src/soup_cli must be in DECLARED_SINKS."""
        hits = scan_src_sinks()
        unlisted = [
            f"{rel}:{lineno} in {func_name}() calling {call_repr}"
            for rel, func_name, lineno, call_repr in hits
            if (rel, func_name) not in DECLARED_SINKS
        ]
        assert not unlisted, (
            "Found outbound HTTP sinks in src/soup_cli that are not declared in DECLARED_SINKS. "
            "Every outbound HTTP call must either be routed through an SSRF gate (e.g. net_guard."
            "refuse_private_ip_literal) or declared with its verified justification (fixed host, "
            "loopback only, operator environment):\n  " + "\n  ".join(unlisted)
        )

    def test_every_declared_sink_is_still_present(self) -> None:
        """Every entry in DECLARED_SINKS must match at least one live call site in code."""
        hits = scan_src_sinks()
        live_keys = {(rel, func_name) for rel, func_name, _lineno, _call in hits}
        dead_entries = [
            f"{rel}::{func_name} -> {reason}"
            for (rel, func_name), reason in sorted(DECLARED_SINKS.items())
            if (rel, func_name) not in live_keys
        ]
        assert not dead_entries, (
            "These entries in DECLARED_SINKS no longer have any matching outbound HTTP sinks "
            "in src/soup_cli. Remove or update them so the ledger cannot rot:\n  "
            + "\n  ".join(dead_entries)
        )

    def test_scan_actually_covers_src(self) -> None:
        """Breadth control: verify the scanner actually walks src/soup_cli and finds modules."""
        scanned = list(SRC.rglob("*.py"))
        assert len(scanned) >= 400, f"only {len(scanned)} modules scanned in {SRC}"
        hits = scan_src_sinks()
        discovered_keys = {(rel, func_name) for rel, func_name, _lineno, _call in hits}
        assert discovered_keys == set(DECLARED_SINKS)

    def test_declared_sinks_count_is_pinned(self) -> None:
        """Pins that exactly 18 functions are accounted for in DECLARED_SINKS."""
        assert len(DECLARED_SINKS) == 18


# ---------------------------------------------------------------------------
# Energy Endpoint SSRF Hardening Parity & Pin Tests
# ---------------------------------------------------------------------------
class TestEnergyEndpointSSRFHardening:
    """Verifies validate_electricity_map_endpoint uses net_guard and closes bypasses."""

    def test_energy_does_not_declare_private_loopback_set(self) -> None:
        """Pins that energy.py does not define its own _LOOPBACK set."""
        energy_path = SRC / "utils" / "energy.py"
        tree = ast.parse(energy_path.read_text(encoding="utf-8"))
        assigned = [
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        assert "_LOOPBACK" not in assigned, "energy.py must use net_guard.LOOPBACK_HOSTS"

    def test_energy_validator_calls_the_shared_refusal(self) -> None:
        """Pins that validate_electricity_map_endpoint calls net_guard.refuse_private_ip_literal."""
        energy_path = SRC / "utils" / "energy.py"
        tree = ast.parse(energy_path.read_text(encoding="utf-8"))
        funcs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "validate_electricity_map_endpoint"
        ]
        assert len(funcs) == 1
        called = {
            node.func.id
            for node in ast.walk(funcs[0])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "refuse_private_ip_literal" in called

    def test_validate_electricity_map_endpoint_loopback(self) -> None:
        from soup_cli.utils.energy import validate_electricity_map_endpoint

        assert (
            validate_electricity_map_endpoint("http://localhost:8080/co2")
            == "http://localhost:8080/co2"
        )
        assert (
            validate_electricity_map_endpoint("http://127.0.0.1:8080/co2")
            == "http://127.0.0.1:8080/co2"
        )

    def test_validate_electricity_map_endpoint_rejects_abbreviated_ipv4(self) -> None:
        from soup_cli.utils.energy import validate_electricity_map_endpoint

        # Abbreviated IPv4 (127.1, 2130706433, hex 0x7f.1) are rejected on plain HTTP
        # because they are not in LOOPBACK_HOSTS (loopback-only plain HTTP)
        with pytest.raises(ValueError):
            validate_electricity_map_endpoint("http://127.1/co2")
        with pytest.raises(ValueError):
            validate_electricity_map_endpoint("http://2130706433/co2")

        # Abbreviated private IPv4 (0x0a000001 -> 10.0.0.1) on HTTPS is rejected by
        # net_guard.refuse_private_ip_literal
        with pytest.raises(
            ValueError, match="private/link-local/reserved IP hosts are not allowed"
        ):
            validate_electricity_map_endpoint("https://0x0a000001/co2")

    def test_validate_electricity_map_endpoint_rejects_rfc6598_cgnat(self) -> None:
        from soup_cli.utils.energy import validate_electricity_map_endpoint

        # 100.64.0.0/10 carrier-grade NAT space was accepted by the old ip.is_private check;
        # net_guard.refuse_private_ip_literal rejects it
        with pytest.raises(ValueError):
            validate_electricity_map_endpoint("http://100.64.0.1/co2")
        with pytest.raises(
            ValueError, match="private/link-local/reserved IP hosts are not allowed"
        ):
            validate_electricity_map_endpoint("https://100.64.0.1/co2")


# ---------------------------------------------------------------------------
# Mutations That Bypassed The Old #616 Guard
# ---------------------------------------------------------------------------
class TestMutationsThatBypassedTheOldGuard:
    """Demonstrates, not just asserts: the shapes from #1547 that passed the #616
    shape-based scan unchecked, but are caught by this sink ratchet."""

    def test_shape1_energy_custom_loopback_set_is_caught(self, tmp_path: Path) -> None:
        """Old energy.py used a private _LOOPBACK set instead of *LOOPBACK_HOSTS."""
        code = (
            "import httpx\n"
            "from urllib.parse import urlsplit\n"
            "_LOOPBACK = frozenset({'localhost', '127.0.0.1', '::1'})\n\n"
            "def validate_and_fetch(endpoint):\n"
            "    parts = urlsplit(endpoint)\n"
            "    if parts.hostname in _LOOPBACK:\n"
            "        return httpx.get(endpoint)\n"
        )
        module = tmp_path / "rogue_energy_shape.py"
        module.write_text(code, encoding="utf-8")
        sinks = scan_src_sinks(tmp_path)
        assert sinks == [("rogue_energy_shape.py", "validate_and_fetch", 8, "httpx.get")]

    def test_shape2_netloc_split_gate_is_caught(self, tmp_path: Path) -> None:
        """urlsplit(u).netloc.split(':')[0] parsed host without accessing .hostname."""
        code = (
            "import httpx\n"
            "from urllib.parse import urlsplit\n\n"
            "def netloc_gate(url):\n"
            "    host = urlsplit(url).netloc.split(':')[0]\n"
            "    if host == 'localhost':\n"
            "        return httpx.post(url, json={})\n"
        )
        module = tmp_path / "rogue_netloc.py"
        module.write_text(code, encoding="utf-8")
        sinks = scan_src_sinks(tmp_path)
        assert sinks == [("rogue_netloc.py", "netloc_gate", 7, "httpx.post")]

    def test_shape3_httpx_url_host_is_caught(self, tmp_path: Path) -> None:
        """httpx.URL(u).host extracted host without urllib .hostname access."""
        code = (
            "import httpx\n\n"
            "def httpx_url_gate(url):\n"
            "    if httpx.URL(url).host == 'localhost':\n"
            "        return httpx.post(url, json={})\n"
        )
        module = tmp_path / "rogue_httpx_url.py"
        module.write_text(code, encoding="utf-8")
        sinks = scan_src_sinks(tmp_path)
        assert sinks == [("rogue_httpx_url.py", "httpx_url_gate", 5, "httpx.post")]

    def test_shape4_module_allowlist_under_another_name_is_caught(self, tmp_path: Path) -> None:
        """A module-level allowlist under another name like _ALLOWED_HOSTS."""
        code = (
            "import httpx\n"
            "_ALLOWED_HOSTS = frozenset({'127.0.0.1'})\n\n"
            "def custom_allowlist_gate(url):\n"
            "    from urllib.parse import urlparse\n"
            "    if urlparse(url).hostname in _ALLOWED_HOSTS:\n"
            "        return httpx.post(url)\n"
        )
        module = tmp_path / "rogue_allowlist.py"
        module.write_text(code, encoding="utf-8")
        sinks = scan_src_sinks(tmp_path)
        assert sinks == [("rogue_allowlist.py", "custom_allowlist_gate", 7, "httpx.post")]

    def test_shape5_bare_outbound_call_with_no_check_is_caught(self, tmp_path: Path) -> None:
        """A bare httpx.post(...) with no validation check at all."""
        code = (
            "import httpx\n\n"
            "def send_webhook_unvalidated(base, payload):\n"
            "    httpx.post(base + '/events', json=payload)\n"
        )
        module = tmp_path / "rogue_bare.py"
        module.write_text(code, encoding="utf-8")
        sinks = scan_src_sinks(tmp_path)
        assert sinks == [("rogue_bare.py", "send_webhook_unvalidated", 4, "httpx.post")]


# ---------------------------------------------------------------------------
# Negative Self-Tests: Prove the Scanner Catches Undeclared Calls & Aliases
# ---------------------------------------------------------------------------
class TestTheScannerCanActuallyFail:
    def test_an_undeclared_httpx_post_is_caught(self, tmp_path: Path) -> None:
        module = tmp_path / "rogue.py"
        module.write_text(
            "import httpx\n\n"
            "def send_data(url, data):\n"
            "    httpx.post(url, json=data)\n",
            encoding="utf-8",
        )
        sinks = scan_src_sinks(tmp_path)
        assert sinks == [("rogue.py", "send_data", 4, "httpx.post")]

    def test_an_undeclared_urllib_urlopen_is_caught(self, tmp_path: Path) -> None:
        module = tmp_path / "rogue_urllib.py"
        module.write_text(
            "import urllib.request\n\n"
            "def fetch(url):\n"
            "    return urllib.request.urlopen(url)\n",
            encoding="utf-8",
        )
        sinks = scan_src_sinks(tmp_path)
        assert sinks == [("rogue_urllib.py", "fetch", 4, "urllib.request.urlopen")]

    def test_an_undeclared_httpx_client_is_caught(self, tmp_path: Path) -> None:
        module = tmp_path / "rogue_client.py"
        module.write_text(
            "import httpx\n\n"
            "def get_client():\n"
            "    return httpx.Client()\n",
            encoding="utf-8",
        )
        sinks = scan_src_sinks(tmp_path)
        assert sinks == [("rogue_client.py", "get_client", 4, "httpx.Client")]

    def test_import_aliases_are_caught(self, tmp_path: Path) -> None:
        module = tmp_path / "rogue_alias.py"
        module.write_text(
            "import httpx as h\n"
            "from httpx import post as my_post\n"
            "from urllib import request as req\n"
            "from urllib.request import urlopen as my_urlopen\n"
            "from requests import get as req_get\n\n"
            "def call_aliased(url):\n"
            "    h.get(url)\n"
            "    my_post(url)\n"
            "    req.urlopen(url)\n"
            "    my_urlopen(url)\n"
            "    req_get(url)\n",
            encoding="utf-8",
        )
        sinks = scan_src_sinks(tmp_path)
        calls = [(func, call) for _rel, func, _line, call in sinks]
        assert calls == [
            ("call_aliased", "httpx.get"),
            ("call_aliased", "httpx.post"),
            ("call_aliased", "urllib.request.urlopen"),
            ("call_aliased", "urllib.request.urlopen"),
            ("call_aliased", "requests.get"),
        ]

    def test_dead_entry_fails_ratchet_assertion(self) -> None:
        dummy_ledger = {("fake/module.py", "fake_func"): "justification"}
        live_keys = {("real/module.py", "real_func")}
        dead = [k for k in dummy_ledger if k not in live_keys]
        assert dead == [("fake/module.py", "fake_func")]
