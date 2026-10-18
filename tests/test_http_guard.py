"""The request guard (src/utils/http_guard.py), against a tiny app of its own.

The checks run Host -> Origin -> token, and every refusal is JSON naming the setting
that would let the request through.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.utils.config_loader import Config
from src.utils.http_guard import (LOOPBACK_HOSTS, TOKEN_ENV, GuardSettings, auth_headers,
                                  guard_settings, install_guard, read_token)

CONSOLE = "http://127.0.0.1:8052"


def _settings(token=None, hosts=(), origins=()):
    return GuardSettings(allowed_hosts=LOOPBACK_HOSTS | set(hosts),
                         allowed_origins=frozenset({CONSOLE, *origins}), token=token)


def _client(settings, exempt=(), base_url="http://127.0.0.1:8051"):
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/")
    def page():
        return {"page": True}

    @app.get("/data")
    def read():
        return {"data": 1}

    @app.post("/data")
    def write():
        return {"written": True}

    install_guard(app, lambda: settings, exempt=exempt)
    return TestClient(app, base_url=base_url)


# ---- the token and the settings ----

@pytest.mark.parametrize("raw, token", [(None, None), ("", None), ("   ", None), (" s3cret ", "s3cret")])
def test_an_empty_token_is_no_token(monkeypatch, raw, token):
    if raw is None:
        monkeypatch.delenv(TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(TOKEN_ENV, raw)
    assert read_token() == token
    assert auth_headers() == ({"Authorization": f"Bearer {token}"} if token else {})


def test_settings_pin_the_console_and_the_service_and_add_the_listed_names(monkeypatch, tmp_path):
    monkeypatch.setenv(TOKEN_ENV, "s3cret")
    cfg = Config({"webui": {"port": 8052},
                  "security": {"allowed_hosts": ["NAS.lan"], "allowed_origins": ["http://nas.lan:8052/"]}},
                 tmp_path)
    s = guard_settings(cfg, own_port=8051)
    assert s.allowed_hosts == LOOPBACK_HOSTS | {"nas.lan"}
    assert s.allowed_origins == {"http://localhost:8051", "http://127.0.0.1:8051",
                                 "http://localhost:8052", "http://127.0.0.1:8052", "http://nas.lan:8052"}
    assert s.token == "s3cret"


def test_a_config_without_the_security_block_allows_loopback_only(monkeypatch, tmp_path):
    monkeypatch.setenv(TOKEN_ENV, "")
    s = guard_settings(Config({}, tmp_path), own_port=8051)
    assert s.allowed_hosts == LOOPBACK_HOSTS and s.token is None
    assert "http://127.0.0.1:8052" in s.allowed_origins       # webui.port's default


def test_a_security_value_that_is_not_a_list_is_an_error(tmp_path):
    cfg = Config({"security": {"allowed_hosts": "nas.lan"}}, tmp_path)
    with pytest.raises(ValueError, match="security.allowed_hosts.*list of strings"):
        guard_settings(cfg, own_port=8051)


def test_a_malformed_security_value_is_a_json_500_naming_the_setting(tmp_path):
    cfg = Config({"security": {"allowed_origins": "http://nas.lan"}}, tmp_path)
    app = FastAPI()
    install_guard(app, lambda: guard_settings(cfg, own_port=8051))
    r = TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False).get("/data")
    assert r.status_code == 500 and r.json()["guard"] == "config"
    assert "ValueError" in r.json()["error"] and "security.allowed_origins" in r.json()["error"]
    assert app.state.guard_settings is None             # read again once the config is fixed


def test_settings_are_read_on_the_first_request_and_kept_until_reset():
    calls = []
    app = FastAPI()

    @app.get("/data")
    def read():
        return {"data": 1}

    install_guard(app, lambda: calls.append(1) or _settings())
    assert calls == [] and app.state.guard_settings is None    # nothing read at install
    client = TestClient(app, base_url="http://127.0.0.1")
    client.get("/data")
    client.get("/data")
    assert calls == [1]
    app.state.guard_settings = None
    client.get("/data")
    assert calls == [1, 1]


def test_the_lifespan_passes_through():
    started = []
    app = FastAPI(on_startup=[lambda: started.append(True)])
    install_guard(app, _settings)
    with TestClient(app, base_url="http://127.0.0.1"):
        pass
    assert started == [True]


# ---- 1. Host ----

@pytest.mark.parametrize("host", ["127.0.0.1:8051", "localhost:8051", "LocalHost", "[::1]:8051"])
def test_loopback_hosts_are_allowed(host):
    assert _client(_settings()).get("/data", headers={"Host": host}).status_code == 200


def test_another_host_is_refused_naming_the_setting_unless_listed():
    r = _client(_settings(), base_url="http://evil.example:8051").get("/health")
    assert r.status_code == 400
    assert r.json() == {"guard": "host", "error": "Host 'evil.example' is not allowed; add it to "
                                                  "security.allowed_hosts in config.yaml"}
    assert _client(_settings(hosts={"nas.lan"}), base_url="http://nas.lan:8051").get("/data").status_code == 200


def test_a_missing_or_malformed_host_is_refused():
    client = _client(_settings())
    assert client.get("/data", headers={"Host": ""}).status_code == 400
    assert client.get("/data", headers={"Host": "[::1"}).status_code == 400


# ---- 2. Origin ----

def test_a_post_from_the_console_origin_or_with_no_origin_passes():
    client = _client(_settings())
    assert client.post("/data", headers={"Origin": CONSOLE}).status_code == 200
    assert client.post("/data").status_code == 200


@pytest.mark.parametrize("origin", ["http://127.0.0.1:8888", "http://localhost:8888", "https://evil.example",
                                    "null"])
def test_a_post_from_another_origin_is_refused_naming_the_setting(origin):
    r = _client(_settings()).post("/data", headers={"Origin": origin})
    assert r.status_code == 403
    assert r.json() == {"guard": "origin", "error": f"Origin {origin!r} may not send POST requests here; "
                                                    "add it to security.allowed_origins in config.yaml"}


def test_a_listed_origin_and_the_request_s_own_origin_pass():
    assert _client(_settings(origins={"http://nas.lan:8052"})).post(
        "/data", headers={"Origin": "http://nas.lan:8052"}).status_code == 200
    # A page this server served itself, whatever port it is published on (a remapped container).
    assert _client(_settings(), base_url="http://127.0.0.1:9052").post(
        "/data", headers={"Origin": "http://127.0.0.1:9052"}).status_code == 200


def test_a_read_is_not_origin_checked():
    assert _client(_settings()).get("/data", headers={"Origin": "https://evil.example"}).status_code == 200


# ---- 3. Token ----

def test_no_token_set_means_no_token_check():
    assert _client(_settings(token=None)).get("/data").status_code == 200


def test_a_set_token_is_required_and_compared():
    client = _client(_settings(token="s3cret"))
    missing = client.get("/data")
    assert missing.status_code == 401 and missing.headers["www-authenticate"] == "Bearer"
    assert missing.json() == {"guard": "token", "error": "missing or wrong bearer token: send "
                              "'Authorization: Bearer <token>' with the value of RAG_API_TOKEN "
                              "from the server's .env"}
    assert client.get("/data", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/data", headers={"Authorization": "Basic s3cret"}).status_code == 401
    assert client.get("/data", headers={"Authorization": "Bearer s3crét".encode()}).status_code == 401
    assert client.get("/data", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/data", headers={"Authorization": "bearer s3cret"}).status_code == 200


def test_health_preflights_and_the_listed_routes_need_no_token():
    client = _client(_settings(token="s3cret"), exempt={("get", "/")})
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
    assert client.options("/data").status_code != 401
    # an exemption is per method, not per path
    assert client.post("/health").status_code == 401


def test_the_checks_run_host_then_origin_then_token():
    s = _settings(token="s3cret")
    assert _client(s, base_url="http://evil.example").post(
        "/data", headers={"Origin": "https://evil.example"}).json()["guard"] == "host"
    assert _client(s).post("/data", headers={"Origin": "https://evil.example"}).json()["guard"] == "origin"


# ---- installed on the two services ----

@pytest.fixture
def console(management_module, monkeypatch, tmp_path):
    """manage_api with the two blind-POST targets stubbed, so a guard that let one
    through would hit a recorder instead of restarting a real :8051 or writing a real
    inbox. The guard's cached settings are reset to read this test's environment."""
    m = management_module
    reached = []
    monkeypatch.setattr(m, "_restart_rag_api", lambda: reached.append("restart") or {"port": 8051})
    monkeypatch.setattr(m, "_inbox", lambda: reached.append("upload") or tmp_path)
    monkeypatch.setattr(m.app.state, "guard_settings", None)
    return TestClient(m.app, base_url="http://127.0.0.1:8052"), reached


@pytest.mark.parametrize("origin", ["https://evil.example", "http://localhost:8888"])
def test_a_foreign_page_can_neither_restart_the_query_api_nor_upload(console, origin):
    client, reached = console
    restart = client.post("/api/service/restart", headers={"Origin": origin})
    upload = client.post("/api/upload", headers={"Origin": origin}, files={"files": ("x.md", b"x")})
    assert (restart.status_code, upload.status_code) == (403, 403)
    assert restart.json()["guard"] == upload.json()["guard"] == "origin"
    assert reached == []
    # the console's own page still can
    assert client.post("/api/service/restart", headers={"Origin": CONSOLE}).status_code == 200
    assert reached == ["restart"]


def test_both_services_refuse_a_host_that_is_not_loopback(console, monkeypatch):
    import serve_api
    monkeypatch.setattr(serve_api.app.state, "guard_settings", None)
    query = TestClient(serve_api.app, base_url="http://rebound.example:8051")
    assert query.get("/health").status_code == 400
    assert TestClient(serve_api.app, base_url="http://127.0.0.1:8051").get("/health").status_code == 200
    client, _ = console
    assert TestClient(client.app, base_url="http://rebound.example:8052").get("/api/schema").status_code == 400


def test_with_a_token_set_the_query_api_401_still_carries_cors_headers(monkeypatch):
    """The console reads :8051 from another port, so a refusal without CORS headers
    would reach its page as an opaque network error instead of a 401 to act on."""
    from types import SimpleNamespace
    import serve_api
    monkeypatch.setenv(TOKEN_ENV, "s3cret")
    monkeypatch.setattr(serve_api.app.state, "guard_settings", None)
    monkeypatch.setitem(serve_api._STATE, "rag", SimpleNamespace(presets={}))
    client = TestClient(serve_api.app, base_url="http://127.0.0.1:8051")
    refused = client.get("/schema", headers={"Origin": CONSOLE})
    assert refused.status_code == 401 and refused.json()["guard"] == "token"
    assert refused.headers["access-control-allow-origin"] == CONSOLE
    assert client.get("/health").status_code == 200                       # probes need no token
    assert client.get("/schema", headers={"Origin": CONSOLE, "Authorization": "Bearer s3cret"}).status_code == 200


def test_with_a_token_set_the_console_page_loads_but_its_api_needs_the_token(console, monkeypatch):
    client, _ = console
    monkeypatch.setenv(TOKEN_ENV, "s3cret")          # before the first request: settings are read then
    assert client.get("/").status_code == 200
    assert client.get("/api/schema").status_code == 401
    assert client.get("/api/schema", headers={"Authorization": "Bearer s3cret"}).status_code == 200
