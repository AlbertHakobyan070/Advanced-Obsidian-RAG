"""fetch-web --backend crawlee: a real fetch backend beside requests / crawl4ai / scrapling.

crawlee is an optional dependency (the `crawlee[playwright,beautifulsoup]`
extras), so every test here swaps a FAKE `crawlee` package into sys.modules: the
contract is pinned without the package or a browser. The fake keeps the real
names and keyword arguments the backend relies on (all checked against crawlee
1.10.3: `Request.from_url(..., always_enqueue=)`, `PlaywrightCrawler(
max_requests_per_crawl=, storage_client=)`, `router.default_handler`,
`failed_request_handler`, `await run([...])`, `await context.page.content()`),
so a misspelt kwarg in the implementation fails here and not only on a machine
that has the package.

Run:  python -m pytest tests/test_web_import_crawlee.py -q
"""
import itertools
import re
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.ingestion import web_import

ROOT = Path(__file__).resolve().parents[1]
URL = "https://example.org/articles/page"
HTML = "<html><body><h1>Title</h1><p>some <b>bold</b> text</p></body></html>"
EXTRAS = "crawlee[playwright,beautifulsoup]"


# ------------------------------------------------------------------- doubles

def _install_fake_crawlee(monkeypatch, pages=None, errors=None, extra_installed=True):
    """Put a fake `crawlee` package into sys.modules and return a spy.

    pages   {url: the html the fake browser renders}
    errors  {url: the exception the fake crawler reports once its retries are spent}
    A URL in neither, or one whose unique_key an earlier crawler already handled,
    is skipped without a word: that is what the real crawler does, because
    crawlee caches its default storages process-wide.
    extra_installed=False models crawlee installed WITHOUT the playwright extra:
    its guarded `PlaywrightCrawler` raises ImportError when it is reached for.
    spy.crawlers holds the keyword arguments of every crawler built, spy.requests
    every request that was run.
    """
    pages, errors = pages or {}, errors or {}
    spy = types.SimpleNamespace(crawlers=[], requests=[])
    handled = set()
    ids = itertools.count()

    class Request:
        def __init__(self, url, unique_key, always_enqueue):
            self.url, self.unique_key, self.always_enqueue = url, unique_key, always_enqueue

        @classmethod
        def from_url(cls, url, *, always_enqueue=False):
            key = f"{next(ids)}|{url}" if always_enqueue else url
            return cls(url, key, always_enqueue)

    class MemoryStorageClient:
        pass

    class PlaywrightCrawler:
        def __init__(self, *, max_requests_per_crawl=None, storage_client=None):
            self.kwargs = {"max_requests_per_crawl": max_requests_per_crawl,
                           "storage_client": storage_client}
            self.router = types.SimpleNamespace(default_handler=self._on_page)
            self._handler = self._failed = None
            spy.crawlers.append(self)

        def _on_page(self, fn):
            self._handler = fn
            return fn

        def failed_request_handler(self, fn):
            self._failed = fn
            return fn

        async def run(self, requests=None):
            for req in requests or []:
                spy.requests.append(req)
                if req.unique_key in handled:
                    continue
                handled.add(req.unique_key)
                if req.url in errors:
                    await self._failed(types.SimpleNamespace(request=req), errors[req.url])
                elif req.url in pages:
                    async def content(html=pages[req.url]):
                        return html
                    await self._handler(types.SimpleNamespace(
                        request=req, page=types.SimpleNamespace(content=content)))

    crawlee = types.ModuleType("crawlee")
    crawlee.Request = Request
    crawlers = types.ModuleType("crawlee.crawlers")
    if extra_installed:
        crawlers.PlaywrightCrawler = PlaywrightCrawler
    else:
        def _guarded(name):
            raise ImportError(f"Install the optional 'playwright' extra to use it: "
                              f"pip install 'crawlee[playwright]' ({name})")
        crawlers.__getattr__ = _guarded
    storage = types.ModuleType("crawlee.storage_clients")
    storage.MemoryStorageClient = MemoryStorageClient
    crawlee.crawlers, crawlee.storage_clients = crawlers, storage
    for name, module in (("crawlee", crawlee), ("crawlee.crawlers", crawlers),
                         ("crawlee.storage_clients", storage)):
        monkeypatch.setitem(sys.modules, name, module)
    spy.MemoryStorageClient = MemoryStorageClient
    return spy


def _make_crawlee_unavailable(monkeypatch, how):
    if how == "not-installed":
        for name in ("crawlee", "crawlee.crawlers", "crawlee.storage_clients"):
            monkeypatch.setitem(sys.modules, name, None)      # `import` of a None entry raises ImportError
    else:
        _install_fake_crawlee(monkeypatch, extra_installed=False)


def _forbid_other_backends(monkeypatch):
    """Touching any other backend is recorded and fails the call."""
    called = []

    def trap(name):
        def _fn(url):
            called.append(name)
            raise AssertionError(f"{name} was used for a crawlee request")
        return _fn

    for name in ("_fetch_requests", "_fetch_crawl4ai", "_fetch_scrapling"):
        monkeypatch.setattr(web_import, name, trap(name))
    return called


# ------------------------------------------------------------------- the backend

def test_crawlee_is_a_registered_backend():
    assert "crawlee" in web_import.FETCH_BACKENDS


def test_backend_returns_the_rendered_page_as_html(monkeypatch):
    _install_fake_crawlee(monkeypatch, pages={URL: HTML})

    payload, ctype = web_import.fetch_one(URL, "crawlee")

    assert payload == HTML.encode("utf-8")
    assert ctype == "text/html"       # not markdown: fetch_urls then runs it through markitdown


def test_rendered_html_goes_through_the_shared_markdown_path(monkeypatch, tmp_path):
    _install_fake_crawlee(monkeypatch, pages={URL: HTML})
    seen = []

    class _MarkItDown:
        def convert(self, path):
            seen.append((Path(path).suffix, Path(path).read_bytes()))
            return types.SimpleNamespace(text_content="# converted by markitdown")

    monkeypatch.setattr(web_import, "_markitdown", lambda: _MarkItDown())

    res = web_import.fetch_urls([URL], tmp_path, backend="crawlee")

    assert res[0]["ok"] is True, res
    assert seen == [(".html", HTML.encode("utf-8"))]       # the same temp-.html hand-off scrapling/requests use
    staged = (tmp_path / res[0]["file"]).read_text(encoding="utf-8")
    assert "# converted by markitdown" in staged and f"> Source: {URL}" in staged


def test_real_markitdown_turns_the_crawlee_html_into_markdown(monkeypatch, tmp_path):
    pytest.importorskip("markitdown")
    _install_fake_crawlee(monkeypatch, pages={URL: HTML})

    res = web_import.fetch_urls([URL], tmp_path, backend="crawlee")

    assert res[0]["ok"] is True, res
    staged = (tmp_path / res[0]["file"]).read_text(encoding="utf-8")
    assert "# Title" in staged and "**bold**" in staged


def test_one_page_is_crawled_into_memory_not_into_a_storage_folder(monkeypatch):
    spy = _install_fake_crawlee(monkeypatch, pages={URL: HTML})

    web_import.fetch_one(URL, "crawlee")

    (crawler,) = spy.crawlers
    assert crawler.kwargs["max_requests_per_crawl"] == 1
    # crawlee's default storage client writes a ./storage folder into the working
    # directory, which for `python main.py fetch-web` is the repo root
    assert isinstance(crawler.kwargs["storage_client"], spy.MemoryStorageClient)
    assert [r.url for r in spy.requests] == [URL]


def test_the_same_url_can_be_fetched_twice_in_one_run(monkeypatch):
    """crawlee caches its storages process-wide, so a plain request for a URL an
    earlier crawler handled is skipped WITHOUT an error and the repeat comes back
    empty (seen against the real package). A link pasted twice must still fetch."""
    spy = _install_fake_crawlee(monkeypatch, pages={URL: HTML})

    first = web_import.fetch_one(URL, "crawlee")
    second = web_import.fetch_one(URL, "crawlee")

    assert first == second == (HTML.encode("utf-8"), "text/html")
    assert all(r.always_enqueue for r in spy.requests)


def test_real_crawlee_still_offers_everything_the_backend_calls():
    """The fakes are only as good as their likeness to the real package. Skipped
    where crawlee is not installed; where it is (the desktop, after a pin bump)
    this fails the day an upgrade renames something the backend relies on."""
    pytest.importorskip("crawlee")
    import dataclasses
    import inspect

    from crawlee import Request
    from crawlee.crawlers import BasicCrawler, PlaywrightCrawler, PlaywrightCrawlingContext
    from crawlee.storage_clients import MemoryStorageClient

    basic_kwargs = set(inspect.signature(BasicCrawler.__init__).parameters)
    assert {"max_requests_per_crawl", "storage_client"} <= basic_kwargs
    assert "always_enqueue" in inspect.signature(Request.from_url).parameters
    assert inspect.iscoroutinefunction(PlaywrightCrawler.run)
    assert callable(PlaywrightCrawler.failed_request_handler)
    assert "page" in {f.name for f in dataclasses.fields(PlaywrightCrawlingContext)}
    crawler = PlaywrightCrawler(max_requests_per_crawl=1, storage_client=MemoryStorageClient())
    assert callable(crawler.router.default_handler)


# ------------------------------------------------------------------- failing readable

@pytest.mark.parametrize("how", ["not-installed", "extra-missing"])
def test_missing_package_raises_runtimeerror_naming_the_extra(monkeypatch, how):
    _make_crawlee_unavailable(monkeypatch, how)

    with pytest.raises(RuntimeError) as err:
        web_import.fetch_one(URL, "crawlee")

    assert EXTRAS in str(err.value)
    assert isinstance(err.value.__cause__, ImportError)    # the real reason is chained, not swallowed


def test_missing_package_is_reported_per_url_and_stages_nothing(monkeypatch, tmp_path):
    _make_crawlee_unavailable(monkeypatch, "not-installed")
    monkeypatch.setattr(web_import, "_markitdown", lambda: object())

    (res,) = web_import.fetch_urls([URL], tmp_path, backend="crawlee")

    assert res["ok"] is False
    assert "RuntimeError" in res["error"] and EXTRAS in res["error"]
    assert list(tmp_path.iterdir()) == []


def test_a_failed_crawl_raises_with_the_crawlee_error_chained(monkeypatch):
    boom = ValueError("Client error status code returned (status code: 404).")
    _install_fake_crawlee(monkeypatch, errors={URL: boom})

    with pytest.raises(RuntimeError) as err:
        web_import.fetch_one(URL, "crawlee")

    assert URL in str(err.value) and "status code: 404" in str(err.value)
    assert err.value.__cause__ is boom


def test_a_crawl_that_returns_no_page_is_an_error_not_an_empty_page(monkeypatch):
    other = "https://example.org/blank"
    _install_fake_crawlee(monkeypatch, pages={other: ""})    # URL itself: neither rendered nor failed

    with pytest.raises(RuntimeError, match="no page"):
        web_import.fetch_one(URL, "crawlee")
    # whereas a page that really rendered to nothing is still a page
    assert web_import.fetch_one(other, "crawlee") == (b"", "text/html")


@pytest.mark.parametrize("scenario", ["not-installed", "crawl-failed", "no-page"])
def test_requesting_crawlee_never_falls_back_to_another_backend(monkeypatch, scenario):
    called = _forbid_other_backends(monkeypatch)
    if scenario == "not-installed":
        _make_crawlee_unavailable(monkeypatch, "not-installed")
    elif scenario == "crawl-failed":
        _install_fake_crawlee(monkeypatch, errors={URL: ValueError("boom")})
    else:
        _install_fake_crawlee(monkeypatch)

    with pytest.raises(RuntimeError):
        web_import.fetch_one(URL, "crawlee")

    assert called == []


# ------------------------------------------------------------------- registration

def test_auto_keeps_its_chain_and_never_picks_crawlee(monkeypatch):
    """crawlee is opt-in: `auto` is still crawl4ai -> scrapling -> requests."""
    spy = _install_fake_crawlee(monkeypatch, pages={URL: HTML})
    order = []

    def not_installed(name):
        def _fn(url):
            order.append(name)
            raise RuntimeError(f"{name} is not installed")
        return _fn

    def plain_requests(url):
        order.append("requests")
        return b"plain", "text/html"

    monkeypatch.setattr(web_import, "_fetch_crawl4ai", not_installed("crawl4ai"))
    monkeypatch.setattr(web_import, "_fetch_scrapling", not_installed("scrapling"))
    monkeypatch.setattr(web_import, "_fetch_requests", plain_requests)

    assert web_import.fetch_one(URL, "auto") == (b"plain", "text/html")
    assert order == ["crawl4ai", "scrapling", "requests"]
    assert spy.crawlers == []


def test_every_list_of_fetch_backends_names_the_same_backends(management_module):
    """The allowed values are spelled out in several places that nothing else
    ties together: the module, the CLI choices, the console's job builder, the
    two schema maps agents read, and the console's backend picker."""
    import main                      # imported here, as test_bench_cli does: it configures logging

    backends = web_import.FETCH_BACKENDS

    parser = main.build_parser()
    for backend in backends:
        args = parser.parse_args(["fetch-web", "--urls", URL, "--backend", backend])
        assert args.backend == backend
    with pytest.raises(SystemExit):
        parser.parse_args(["fetch-web", "--urls", URL, "--backend", "nope"])

    for backend in backends:
        argv = management_module._build_argv("fetch_web", {"urls": [URL], "backend": backend})
        assert argv[argv.index("--backend") + 1] == backend
    with pytest.raises(ValueError):
        management_module._build_argv("fetch_web", {"urls": [URL], "backend": "nope"})

    schema = management_module.api_schema()
    spelled = "|".join(backends)
    assert schema["endpoints"]["POST /api/import/fetch"]["body"]["backend"] == spelled
    assert schema["job_kinds"]["fetch_web"]["params"]["backend"] == spelled

    html = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    picker = html[html.index('id="fx-backend"'):]
    picker = picker[:picker.index("</select>")]
    assert re.findall(r'<option value="([^"]+)"', picker) == list(backends)
