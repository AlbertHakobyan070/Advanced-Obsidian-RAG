"""The console's OCR and Graph tabs (webui/index.html): in the nav, wired only to
routes that exist, with their keyboard shortcuts guarded and their saves fenced.

Grep-style on purpose, like the other tests that pin the console's markup: the page
cannot run without a browser. Both tabs were driven end to end with Playwright
against a spare-port console on a scratch vault; the API side has its own tests
(test_ocr_studio.py, test_canvas_edit.py).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
API = (ROOT / "manage_api.py").read_text(encoding="utf-8")


def _script(banner: str, until: str) -> str:
    start = HTML.index(banner)
    return HTML[start:HTML.index(until, start)]


def _ocr() -> str:
    return _script("/* ---------- OCR Studio (OCR tab)", "/* ---------- Graph tab")


def _graph() -> str:
    return _script("/* ---------- Graph tab: one Obsidian canvas", "/* ---------- boot ---------- */")


def _fn(js: str, name: str) -> str:
    """One function's source: from its header to the next top-level function."""
    start = re.search(rf"(async )?function {name}\(", js).start()
    nxt = re.search(r"\n(async )?function \w+\(|\n/\* ", js[start + 1:])
    return js[start:start + 1 + nxt.start()] if nxt else js[start:]


def test_the_nav_has_both_tabs_and_opening_one_loads_it():
    for tab, label, opener in (("ocr", "OCR", "osOpen"), ("graph", "Graph", "gxOpen")):
        assert f'<button data-tab="{tab}">{label}</button>' in HTML
        assert f'<section id="tab-{tab}">' in HTML
        assert f'if(b.dataset.tab==="{tab}"){opener}();' in HTML


def test_every_route_the_tabs_call_exists_in_the_console():
    for js, prefix in ((_ocr(), "/api/ocrstudio/"), (_graph(), "/api/canvas/")):
        called = set(re.findall(re.escape(prefix) + r"[a-z_.]+", js))
        assert called, prefix
        for route in called:
            assert re.search(rf'@app\.(get|post)\("{re.escape(route)}"\)', API), route


def test_the_ocr_tab_calls_its_whole_workflow():
    js = _ocr()
    for route in ("engines", "files", "doc", "page", "page.png", "mask", "mask.png",
                  "propose", "ocr", "ingest", "quality", "upload", "pageset", "export"):
        assert f"/api/ocrstudio/{route}" in js, route


def test_imports_go_one_file_per_request_and_the_strip_follows_the_page_set():
    js = _ocr()
    imp = _fn(js, "osImport")
    assert '"/api/ocrstudio/upload"' in imp and 'fd.append("file"' in imp and "for(" in imp
    assert "/api/upload" not in js                    # never the Ingest tab's inbox upload
    assert "in_set" in _fn(js, "osRenderStrip")
    assert "in_set" in _fn(js, "osOpenDoc")           # the first page opened is a working page
    # the export is a download of the server's zip, not a JSON call
    exp = _fn(js, "osExport")
    assert 'authFetch("/api/ocrstudio/export"' in exp and "URL.createObjectURL" in exp


def test_shortcuts_in_both_tabs_ignore_typing_and_other_tabs():
    for js, tab in ((_ocr(), "tab-ocr"), (_graph(), "tab-graph")):
        h = js[js.index('document.addEventListener("keydown"'):]
        h = h[:h.index("\n});")]
        assert f'$("#{tab}").classList.contains("on")' in h        # only while the tab is shown
        assert 'closest("input,textarea,select")' in h             # a letter in a field is text
        # Ctrl+S saves even from a text box; nothing else does
        assert h.index('key==="s"') < h.index("if(typing")
        assert h.index("if(typing") < h.index('key==="delete"')


def test_the_type_palette_only_picks_the_next_regions_type():
    body = _fn(_ocr(), "osPickType")
    assert "state.os.type=t" in body
    assert "state.os.sel" not in body and "osSetType" not in body


def test_ocr_markdown_is_sanitized_after_rendering():
    body = _fn(_ocr(), "osMd")
    assert "marked.parse(text)" in body
    assert 'querySelectorAll("script,style,iframe,object,embed,link,meta,form,base")' in body
    assert "/^on/i.test(a.name)" in body                       # inline handlers stripped


def test_the_page_image_is_never_served_from_the_image_cache():
    # the same URL after a save (heal, deskew) would come back unhealed
    assert "p.t=Date.now()" in _fn(_ocr(), "osImgUrl")


def test_proposals_and_ocr_run_on_the_saved_page():
    js = _ocr()
    for fn in ("osPropose", "osRun", "osIngest"):
        assert "await osFlush()" in _fn(js, fn), fn


def test_graph_save_is_explicit_and_carries_the_loaded_hash():
    js = _graph()
    save = _fn(js, "gxSave")
    assert 'authFetch("/api/canvas/file",{method:"POST"' in save
    assert "base_sha256:state.gx.sha" in save
    assert "r.status===409" in save and 'data-gx="reload"' in save
    # the only writes to the canvas file are in gxSave
    assert js.count('"/api/canvas/file",{method:"POST"') == 1
    # a suggested label only fills the box; it never saves
    assert "/api/canvas/file" not in _fn(js, "gxSuggest")


def test_reindex_reads_the_file_on_disk_so_it_refuses_unsaved_edits():
    body = _fn(_graph(), "gxReindex")
    assert body.index("if(state.gx.dirty)") < body.index('osPost("/api/canvas/reindex"')
    # asked in the console's own dialog, and nothing is posted before the answer comes back
    assert body.index("await uiConfirm(") < body.index('osPost("/api/canvas/reindex"')


def test_leaving_the_page_with_unsaved_edits_asks_first():
    assert 'window.addEventListener("beforeunload",e=>{if(state.gx.dirty||state.os.dirty)' in HTML


# ---- suggest label v2, label tags, removal from the index, graph embedder, the OCR queue x

def test_an_api_error_keeps_the_whole_response_body():
    # The owner's opaque errors were 502s whose body (the stage, the model's reply) api() dropped.
    body = _fn(HTML, "api")
    assert ".body=j" in body and ".status=r.status" in body


def test_suggest_sends_the_edge_and_the_unsaved_canvas():
    js = _graph()
    body = _fn(js, "gxSuggest")
    assert "edge_id:" in body and "data:state.gx.data" in body      # unsaved edits count
    assert "from_text" not in body and "gxText(" not in js          # the v1 body is gone


def test_a_failed_suggestion_shows_the_models_reply():
    js = _graph()
    caught = _fn(js, "gxSuggest")
    caught = caught[caught.index("catch(err)"):]
    assert "err.body" in caught and "reply" in caught and "stage" in caught
    view = _fn(js, "gxSuggestHtml")
    assert "esc(l.reply)" in view and "esc(l.stage)" in view and 'role="alert"' in view


def test_suggested_tags_go_into_the_label_as_hashtags_with_undo():
    js = _graph()
    for fn in ("gxApplyLabel", "gxToggleTag"):
        body = _fn(js, fn)
        assert body.index("gxPushUndo()") < body.index("e.label="), fn     # undo first, then the edit
        assert "gxSyncLabel(e)" in body, fn
    assert '"#"+tag' in _fn(js, "gxToggleTag")
    sync = _fn(js, "gxSyncLabel")
    assert "gxMarkDirty()" in sync and "gxSchedule()" in sync and "aria-pressed" in sync
    assert 'aria-pressed="${gxHasTag(' in _fn(js, "gxSuggestHtml")     # a chip says if its tag is in


def test_remove_confirms_before_deleting_and_never_writes_the_canvas():
    js = _graph()
    body = _fn(js, "gxRemove")
    assert body.index("await uiConfirm(") < body.index('osPost("/api/documents/delete"')
    assert "rebuild:true" in body
    assert body.index('osPost("/api/documents/delete"') < body.index('kind:"graph_embed"')
    assert '"/api/canvas/file",{method:"POST"' not in body          # the .canvas file is never written
    assert 'data-gx="restart"' in body                               # :8051 still holds the canvas
    restart = _fn(js, "gxRestart")
    assert restart.index('"/api/jobs/"') < restart.index('$("#rag-restart").click()')   # the header's flow


def test_the_embedder_build_writes_the_setting_before_queueing():
    body = _fn(_graph(), "gxEmbBuild")
    assert (body.index("await uiConfirm(") < body.index('osPost("/api/settings"')
            < body.index('kind:"graph_embed"'))
    assert '"graph.embedding_provider":v' in body


def test_only_queue_rows_carry_the_remove_button_and_it_asks_first():
    js = _ocr()
    rows = _fn(js, "osLoadFiles")
    assert 'sec("imported",r.uploads,null,true)' in rows
    assert re.search(r'sec\("inbox",r\.inbox,[^)]*,true\)', rows)
    assert 'sec("worked on",r.studied)' in rows                       # not in the queue: no x
    assert re.search(r'sec\(`search: [^`]*`,r\.rows,"[^"]*"\)', rows)  # nor are search hits
    assert 'title="Remove from the OCR queue"' in rows and "aria-label=" in rows
    rm = _fn(js, "osRemove")
    assert rm.index("await uiConfirm(") < rm.index('osPost("/api/ocrstudio/remove"')
    assert "danger:true" not in rm                 # nothing is deleted: the file moves to .trash
    assert "e.status===409" in rm and "osCloseDoc()" in rm


def test_deleting_a_staged_conversion_asks_first():
    # Unlike the OCR tab's x, the Ingest tab's x on a _converted file deletes it from disk for good
    body = _fn(HTML, "loadConverted")
    handler = body[body.index('$$(".fx-del")'):]
    post = handler.index('api("/api/inbox/delete"')
    assert handler.index("await uiConfirm(") < post
    assert "danger:true" in handler[:post] and "cannot be restored" in handler[:post]
    # each x names its folder: an inbox original may share the staged file's name
    assert 'folder:"converted"' in handler[post:post + 300]
    inbox = HTML[HTML.index('$$(".ibx-del")'):]
    assert 'folder:"inbox"' in inbox[:inbox.index("loadInbox()")]
