"""_print_pdf prints a page that never goes network-idle (ads, analytics,
polling) instead of failing it after the 90 s goto timeout. A fake
playwright.sync_api stands in for Chromium: no browser, no network."""
import sys
import types

from src.ingestion import web_import


class _Timeout(Exception):
    pass


class _Page:
    def __init__(self, calls):
        self.calls = calls

    def goto(self, url, wait_until, timeout):
        self.calls.append(("goto", wait_until))

    def wait_for_load_state(self, state, timeout):
        self.calls.append(("idle", state))
        raise _Timeout("never idle")

    def wait_for_timeout(self, ms):
        pass

    def pdf(self, path, **kw):
        self.calls.append(("pdf",))
        open(path, "wb").write(b"%PDF-fake")


def test_a_page_that_never_goes_idle_is_still_printed(tmp_path, monkeypatch):
    calls = []
    browser = types.SimpleNamespace(new_page=lambda **kw: _Page(calls), close=lambda: None)
    pw = types.SimpleNamespace(chromium=types.SimpleNamespace(launch=lambda: browser))

    class _Ctx:
        def __enter__(self):
            return pw

        def __exit__(self, *a):
            return False

    fake = types.ModuleType("playwright.sync_api")
    fake.Error, fake.TimeoutError, fake.sync_playwright = Exception, _Timeout, _Ctx
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake)
    dest = tmp_path / "page.pdf"
    web_import._print_pdf("https://example.invalid/busy", dest)
    assert calls == [("goto", "load"), ("idle", "networkidle"), ("pdf",)]
    assert dest.read_bytes() == b"%PDF-fake"
