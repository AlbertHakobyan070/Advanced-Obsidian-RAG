"""The console page's API-token plumbing (webui/index.html).

Grep-style on purpose, like the other tests that pin the console's markup: the page
cannot run without a browser. With RAG_API_TOKEN set on the server, a request that
skips authFetch() is a 401 the page never recovers from, and a load the browser makes
itself that skips authedUrl() is a broken image, frame or download. These keep every
call site on the helpers, and the token out of every URL.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
BARE_FETCH = re.compile(r"(?<![\w.])fetch\(")
NODE = shutil.which("node")


def _function(name: str) -> str:
    """One top-level `function name(` or `async function name(`, through its closing brace."""
    start = re.search(rf"(?m)^(?:async )?function {name}\(", HTML).start()
    return HTML[start:HTML.index("\n}\n", start) + 2]


def test_the_helpers_exist_and_keep_the_token_in_the_ledger_namespace():
    assert 'const TOKEN_KEY="ledger-api-token";' in HTML
    assert "function authHeaders(" in HTML
    assert 'Authorization:"Bearer "+t' in _function("authHeaders")
    fetcher = _function("authFetch")
    # the console's own dialog asks, never the browser's native prompt (which ignores the theme)
    assert "r.status!==401" in fetcher and "uiPrompt(" in fetcher
    assert not re.search(r"(?<![\w.])prompt\(", fetcher)
    # only the token guard's 401 asks for the token, and the caller still gets an unread body
    assert 'why.guard!=="token"' in fetcher and "r.clone().json()" in fetcher
    assert fetcher.index('why.guard!=="token"') < fetcher.index("uiPrompt(")
    assert "localStorage.setItem(TOKEN_KEY,t)" in fetcher


def test_every_request_goes_through_authfetch():
    fetcher = _function("authFetch")
    assert BARE_FETCH.search(fetcher)                         # the one place that calls fetch()
    stray = [ln.strip() for ln in HTML.replace(fetcher, "").splitlines() if BARE_FETCH.search(ln)]
    assert stray == [], "requests that would not carry the API token"
    assert "await authFetch(path,opts)" in _function("api")
    assert "authFetch(state.ragApi+path,opts)" in _function("ragCall")
    assert "await api(path,opts)" in _function("evApi")       # evApi / evPost ride on api()


def test_the_four_header_less_loads_go_through_authedurl():
    assert "if(!apiToken())return path;" in _function("authedUrl")   # no token: the plain URL, as before
    assert len(re.findall(r"(?<!function )authedUrl\(", HTML)) == 4
    preview = _function("previewFile")
    assert "src=await authedUrl(url)" in preview and 'iframe src="${esc(src)}"' in preview
    assert "href=await authedUrl(u)" in HTML and "a.href=u;" not in HTML
    image = _function("osLoadImage")
    assert "authedUrl(url).then(" in image and "img.src=url" not in image
    mask = _function("osSetupMask")
    assert "authedUrl(mp).then(" in mask and 'm.src="/api/' not in mask


def test_the_token_never_goes_into_a_url_and_can_be_forgotten():
    assert HTML.count('"Bearer "') == 1                       # built in authHeaders() only
    assert not re.search(r"[?&]token=|TOKEN_KEY\)\+|apiToken\(\)\+", HTML)
    assert 'id="token-forget"' in HTML
    handler = re.search(r'\$\("#token-forget"\)\.onclick=.*', HTML).group(0)
    assert "localStorage.removeItem(TOKEN_KEY)" in handler and "tokenDeclined=false" in handler


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_concurrent_401s_raise_one_token_prompt_and_all_retry_on_its_answer():
    """The prompt is async now (a themed dialog, not the blocking native one), so two requests
    refused while it is open must share it: asked once, both retried with the answer. A refusal
    is remembered, so the next 401 does not ask again either."""
    start = HTML.index('const TOKEN_KEY="ledger-api-token";')
    block = HTML[start:HTML.index("\n}\n", HTML.index("async function authFetch(")) + 2]
    harness = """
      const store = new Map(), waiting = [], sent = [];
      const localStorage = {getItem: k => store.has(k) ? store.get(k) : null,
                            setItem: (k, v) => store.set(k, String(v)), removeItem: k => store.delete(k)};
      let asks = 0;
      const uiPrompt = () => { asks++; return new Promise(res => waiting.push(res)); };
      const answer = v => waiting.splice(0).forEach(res => res(v));
      const resp = (status, body) => ({status, clone() { return this; }, json: async () => body});
      const fetch = async (url, opts) => {
        const auth = (opts.headers || {}).Authorization || "";
        sent.push(auth);
        if (url === "/other") return resp(401, {});                    // a 401 the token guard did not send
        return auth === "Bearer good" ? resp(200, {ok: true}) : resp(401, {guard: "token"});
      };
      const load = () => new Function("localStorage", "fetch", "uiPrompt", __BLOCK__ +
        "; return {authFetch, declined: () => tokenDeclined};")(localStorage, fetch, uiPrompt);
      const tick = () => new Promise(r => setTimeout(r, 0));
      (async () => {
        const out = {};
        let p = load();
        const a = p.authFetch("/a", {}), b = p.authFetch("/b", {});
        await tick();
        out.open = {asks, waiting: waiting.length};
        answer("  good ");
        out.ok = {statuses: (await Promise.all([a, b])).map(r => r.status), asks, stored: store.get("ledger-api-token")};
        out.after = {status: (await p.authFetch("/c", {})).status, asks};
        out.not_the_guard = {status: (await p.authFetch("/other", {})).status, asks};
        store.clear(); asks = 0; p = load();
        const c = p.authFetch("/a", {}), d = p.authFetch("/b", {});
        await tick(); answer(null);
        out.declined = {statuses: (await Promise.all([c, d])).map(r => r.status), asks, flag: p.declined()};
        out.declined.later = {status: (await p.authFetch("/e", {})).status, asks};
        console.log(JSON.stringify(out));
      })();
    """.replace("__BLOCK__", json.dumps(block))
    r = subprocess.run([NODE, "-e", harness], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    o = json.loads(r.stdout)
    assert o["open"] == {"asks": 1, "waiting": 1}                         # two 401s, one dialog
    assert o["ok"] == {"statuses": [200, 200], "asks": 1, "stored": "good"}
    assert o["after"] == {"status": 200, "asks": 1}                       # the stored token rides along
    assert o["not_the_guard"] == {"status": 401, "asks": 1}               # no token can fix that one
    assert o["declined"] == {"statuses": [401, 401], "asks": 1, "flag": True,
                             "later": {"status": 401, "asks": 1}}
