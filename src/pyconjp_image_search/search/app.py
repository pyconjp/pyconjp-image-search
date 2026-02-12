"""Gradio application for image search UI."""

import json
import re
import tempfile
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import gradio as gr
from PIL import Image

from pyconjp_image_search.config import FLICKR_USER_ID, SIGLIP_MODEL_NAME
from pyconjp_image_search.db import get_connection
from pyconjp_image_search.models import ImageMetadata
from pyconjp_image_search.search.query import (
    get_event_names,
    get_image_embedding,
    search_images_by_text,
)

PAGE_SIZE = 20

# Flickr static URL size suffix pattern: {id}_{secret}_{size}.jpg
_FLICKR_SIZE_RE = re.compile(r"(_[a-z0-9])\.jpg$", re.IGNORECASE)


def _flickr_url_resize(url: str, size: str = "z") -> str:
    """Swap the size suffix in a Flickr static URL.

    Size suffixes: s=75sq, q=150sq, t=100, m=240, z=640, b=1024, h=1600, k=2048
    """
    return _FLICKR_SIZE_RE.sub(f"_{size}.jpg", url)


CUSTOM_CSS = """
.full-height-gallery .grid-wrap {
    overflow-y: visible !important;
    max-height: none !important;
}
.full-height-gallery .fixed-height {
    max-height: none !important;
    min-height: none !important;
}
.full-height-gallery .grid-container {
    grid-template-rows: none !important;
}
.thumb-strip .grid-wrap {
    overflow-x: auto !important;
    overflow-y: hidden !important;
    max-height: 120px !important;
}
.thumb-strip .grid-container {
    grid-template-rows: 1fr !important;
    grid-auto-flow: column !important;
    grid-auto-columns: 100px !important;
    grid-template-columns: none !important;
}
/* Crop selection overlay */
.crop-selection {
    position: absolute;
    border: 2px dashed rgba(255,255,255,0.9);
    background: rgba(255,255,255,0.05);
    pointer-events: none;
    z-index: 10;
    box-shadow: 0 0 0 9999px rgba(0,0,0,0.45);
}
.crop-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    cursor: crosshair;
    z-index: 9;
}
"""

# ── Crop tool JS (injected once via gr.HTML) ─────────────────────────
CROP_TOOL_SCRIPT = """
<script>
window._cropStates = {};

window._toast = function(msg, ok) {
    var t = document.createElement('div');
    t.textContent = msg;
    t.style.cssText = 'position:fixed;top:20px;left:50%;transform:translateX(-50%);'
        + 'background:' + (ok ? '#333' : '#c00') + ';color:#fff;padding:12px 24px;'
        + 'border-radius:8px;z-index:9999;font-size:14px;box-shadow:0 2px 8px rgba(0,0,0,.3);';
    document.body.appendChild(t);
    setTimeout(function(){ t.remove(); }, 2000);
};

/* Compute the actual rendered image rect inside an object-fit:contain element */
function _getRenderedRect(img) {
    var ir = img.getBoundingClientRect();
    var natW = img.naturalWidth;
    var natH = img.naturalHeight;
    if (!natW || !natH) return ir;
    var style = window.getComputedStyle(img);
    var fit = style.objectFit || 'fill';
    if (fit === 'contain' || fit === 'scale-down') {
        var ratio = Math.min(ir.width / natW, ir.height / natH);
        var rw = natW * ratio;
        var rh = natH * ratio;
        return {
            left: ir.left + (ir.width - rw) / 2,
            top: ir.top + (ir.height - rh) / 2,
            width: rw,
            height: rh
        };
    }
    return { left: ir.left, top: ir.top, width: ir.width, height: ir.height };
}

/* Enable / disable the crop-dependent buttons */
function _setCropBtns(containerId, enabled) {
    var prefix = (containerId === 'text-preview') ? 'text' : 'img';
    ['search-cropped-btn', 'copy-clipboard-btn'].forEach(function(suffix) {
        var el = document.getElementById(prefix + '-' + suffix);
        if (!el) return;
        /* Gradio 6: elem_id may be on the <button> itself or on a wrapper */
        var btn = (el.tagName === 'BUTTON') ? el : el.querySelector('button');
        if (!btn) btn = el;
        btn.disabled = !enabled;
        btn.style.opacity = enabled ? '' : '0.4';
        btn.style.pointerEvents = enabled ? '' : 'none';
    });
}

/* Set up the rectangle-selection overlay on the preview image */
function _setupOverlay(container, img, containerId) {
    var wrapper = img.parentElement;
    if (!wrapper) return;
    wrapper.style.position = 'relative';

    var overlay = document.createElement('div');
    overlay.className = 'crop-overlay';
    var sel = document.createElement('div');
    sel.className = 'crop-selection';
    sel.style.display = 'none';
    wrapper.appendChild(overlay);
    wrapper.appendChild(sel);

    var st = {active:false, sx:0, sy:0, ex:0, ey:0, has:false};
    window._cropStates[containerId] = st;

    /* Disable buttons until a rectangle is drawn */
    _setCropBtns(containerId, false);

    overlay.addEventListener('mousedown', function(e) {
        e.preventDefault();
        var rr = _getRenderedRect(img);
        var x = e.clientX - rr.left;
        var y = e.clientY - rr.top;
        if (x < 0 || y < 0 || x > rr.width || y > rr.height) return;
        st.active = true;
        st.sx = x; st.sy = y; st.ex = x; st.ey = y;
        st.has = false;
        sel.style.display = 'block';
        _updateSel(sel, st, img);
        _setCropBtns(containerId, false);
    });
    document.addEventListener('mousemove', function(e) {
        if (!st.active) return;
        var rr = _getRenderedRect(img);
        st.ex = Math.max(0, Math.min(e.clientX - rr.left, rr.width));
        st.ey = Math.max(0, Math.min(e.clientY - rr.top, rr.height));
        _updateSel(sel, st, img);
    });
    document.addEventListener('mouseup', function() {
        if (!st.active) return;
        st.active = false;
        var w = Math.abs(st.ex - st.sx), h = Math.abs(st.ey - st.sy);
        st.has = (w > 5 && h > 5);
        if (!st.has) sel.style.display = 'none';
        _setCropBtns(containerId, st.has);
    });
}

/* Position the selection div accounting for image offset inside wrapper */
function _updateSel(el, s, img) {
    var wr = img.parentElement.getBoundingClientRect();
    var rr = _getRenderedRect(img);
    var offX = rr.left - wr.left;
    var offY = rr.top  - wr.top;
    var x = Math.min(s.sx, s.ex) + offX;
    var y = Math.min(s.sy, s.ey) + offY;
    var w = Math.abs(s.ex - s.sx);
    var h = Math.abs(s.ey - s.sy);
    el.style.left = x + 'px'; el.style.top = y + 'px';
    el.style.width = w + 'px'; el.style.height = h + 'px';
}

window.initCrop = function(containerId, retries) {
    retries = retries || 0;
    var container = document.getElementById(containerId);
    if (!container) return;

    /* Clean up previous overlay */
    container.querySelectorAll('.crop-overlay,.crop-selection')
             .forEach(function(e){ e.remove(); });

    var img = container.querySelector('img');
    if (!img || !img.src) {
        if (retries < 15) {
            setTimeout(function() { window.initCrop(containerId, retries + 1); }, 200);
        }
        return;
    }

    if (img.complete && img.naturalWidth > 0) {
        _setupOverlay(container, img, containerId);
    } else {
        img.addEventListener('load', function onload() {
            img.removeEventListener('load', onload);
            _setupOverlay(container, img, containerId);
        });
    }
};

/* Get crop data as JSON string (image URL + crop rect in natural pixels) */
window.getCropData = function(containerId) {
    var container = document.getElementById(containerId);
    if (!container) return null;
    var img = container.querySelector('img');
    if (!img) return null;
    var st = window._cropStates[containerId];
    if (!st || !st.has) return null;

    var rr = _getRenderedRect(img);
    var scX = img.naturalWidth  / rr.width;
    var scY = img.naturalHeight / rr.height;

    return JSON.stringify({
        url: img.src,
        x: Math.round(Math.min(st.sx, st.ex) * scX),
        y: Math.round(Math.min(st.sy, st.ey) * scY),
        w: Math.round(Math.abs(st.ex - st.sx) * scX),
        h: Math.round(Math.abs(st.ey - st.sy) * scY)
    });
};
</script>
"""

# ── JS snippets for Gradio events ────────────────────────────────────

SCROLL_AND_INIT_CROP_JS = """
(args) => {
    setTimeout(() => {
        const el = document.getElementById('%s');
        if (el) {
            el.scrollIntoView({behavior: 'smooth', block: 'start'});
            setTimeout(() => { if (window.initCrop) window.initCrop('%s'); }, 300);
        }
    }, 100);
}
"""

REINIT_CROP_JS = """
(args) => {
    setTimeout(() => { if (window.initCrop) window.initCrop('%s'); }, 300);
}
"""

COPY_CLIPBOARD_JS = """
async () => {
    const data = window.getCropData('%s');
    if (!data) { window._toast('Select an area first', false); return; }
    const d = JSON.parse(data);
    try {
        /* Load a separate CORS-enabled image for canvas operations */
        const corsImg = new window.Image();
        corsImg.crossOrigin = 'anonymous';
        await new Promise((resolve, reject) => {
            corsImg.onload = resolve;
            corsImg.onerror = () => reject(new Error('CORS'));
            corsImg.src = d.url;
        });
        const canvas = document.createElement('canvas');
        canvas.width = d.w;
        canvas.height = d.h;
        canvas.getContext('2d').drawImage(corsImg, d.x, d.y, d.w, d.h, 0, 0, d.w, d.h);
        const blob = await new Promise(r => canvas.toBlob(r, 'image/png'));
        await navigator.clipboard.write([new ClipboardItem({'image/png': blob})]);
        window._toast('Cropped area copied!', true);
    } catch (e) {
        console.error('Clipboard copy failed:', e);
        window._toast('Failed to copy to clipboard', false);
    }
}
"""

CROP_TO_JSON_JS = """
() => {
    const data = window.getCropData('%s');
    return data || '';
}
"""


def create_app() -> gr.Blocks:
    """Create and return the Gradio Blocks app."""
    conn = get_connection()

    event_names = get_event_names(conn)

    # Lazy-loaded embedder
    _embedder_cache: dict = {}

    def _get_embedder():
        if "instance" not in _embedder_cache:
            from pyconjp_image_search.embedding.siglip import SigLIPEmbedder

            _embedder_cache["instance"] = SigLIPEmbedder(model_name=SIGLIP_MODEL_NAME)
        return _embedder_cache["instance"]

    def _make_gallery_items(
        results: list[tuple[ImageMetadata, float]],
    ) -> tuple[list[tuple[str, str]], list[ImageMetadata]]:
        """Build gallery items and metadata list from search results."""
        items = []
        metadata = []
        for meta, score in results:
            url = _flickr_url_resize(meta.image_url, "z")  # 640px for gallery
            caption = f"score: {score:.3f} | {meta.event_name}"
            items.append((url, caption))
            metadata.append(meta)
        return items, metadata

    def _get_preview_url(gallery_item):
        url = gallery_item[0] if isinstance(gallery_item, (list, tuple)) else gallery_item
        return _flickr_url_resize(url, "b") if "staticflickr.com" in url else url

    # ── Preview helpers ──────────────────────────────────────────────

    def _build_preview_caption(gallery_item, metadata_list, index):
        caption = gallery_item[1] if isinstance(gallery_item, (list, tuple)) and len(gallery_item) > 1 else ""
        if index is not None and index < len(metadata_list):
            meta = metadata_list[index]
            if meta.flickr_photo_id and FLICKR_USER_ID:
                page_url = f"https://www.flickr.com/photos/{FLICKR_USER_ID}/{meta.flickr_photo_id}/"
                caption += f" | [Flickr で開く]({page_url})"
        return caption

    def _on_gallery_select(gallery_items: list, metadata_list: list, evt: gr.EventData):
        index = evt._data.get("index")
        if index is not None and gallery_items:
            item = gallery_items[index]
            caption = _build_preview_caption(item, metadata_list, index)
            preview_url = _get_preview_url(item)
            return (
                gr.update(value=preview_url, visible=True),
                gr.update(value=caption, visible=True),
                gr.update(value=gallery_items, visible=True),
                gr.update(visible=True),   # close btn
                gr.update(visible=True),   # find similar btn
                gr.update(visible=True),   # search cropped btn (JS will disable)
                gr.update(visible=True),   # copy clipboard btn (JS will disable)
                index,
            )
        hidden = gr.update(visible=False)
        return (hidden, hidden, hidden, hidden, hidden, hidden, hidden, None)

    def _on_thumb_select(gallery_items: list, metadata_list: list, evt: gr.EventData):
        index = evt._data.get("index")
        if index is not None and gallery_items:
            item = gallery_items[index]
            caption = _build_preview_caption(item, metadata_list, index)
            preview_url = _get_preview_url(item)
            return gr.update(value=preview_url), gr.update(value=caption), index
        return gr.update(), gr.update(), None

    def _on_close_preview():
        return (
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
            gr.update(value=None, visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    # ── Cross-tab handlers ───────────────────────────────────────────

    _noop_10 = tuple(gr.update() for _ in range(10))

    def _do_find_similar(
        selected_index: int | None,
        metadata_list: list,
        selected_events: list[str],
    ) -> tuple:
        if selected_index is None or not metadata_list:
            return _noop_10
        meta = metadata_list[selected_index]
        emb = get_image_embedding(conn, meta.id, SIGLIP_MODEL_NAME)
        if emb is None:
            return _noop_10
        results = search_images_by_text(
            conn,
            query_embedding=emb,
            model_name=SIGLIP_MODEL_NAME,
            limit=PAGE_SIZE,
            offset=0,
            event_names=selected_events or None,
        )
        gallery_items, new_metadata = _make_gallery_items(results)
        has_more = len(results) == PAGE_SIZE
        source_url = _flickr_url_resize(meta.image_url, "b")
        msg = f"Found {len(results)} images similar to selected image."
        return (
            gallery_items, msg, PAGE_SIZE, gallery_items,
            new_metadata, emb.tolist(),
            gr.update(visible=has_more), selected_events,
            source_url, gr.Tabs(selected=1),
        )

    def _do_search_cropped(
        crop_json: str,
        selected_events: list[str],
    ) -> tuple:
        if not crop_json:
            return _noop_10
        data = json.loads(crop_json)
        url = data["url"]
        x, y, w, h = data["x"], data["y"], data["w"], data["h"]

        # Download image from Flickr URL
        response = urllib.request.urlopen(url)  # noqa: S310
        img = Image.open(BytesIO(response.read())).convert("RGB")

        # Crop the selected region
        cropped = img.crop((x, y, x + w, y + h))

        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cropped.save(tmp, format="PNG")
        tmp.close()
        image_path = Path(tmp.name)

        embedder = _get_embedder()
        query_emb = embedder.embed_images([image_path])
        results = search_images_by_text(
            conn,
            query_embedding=query_emb,
            model_name=SIGLIP_MODEL_NAME,
            limit=PAGE_SIZE,
            offset=0,
            event_names=selected_events or None,
        )
        gallery_items, new_metadata = _make_gallery_items(results)
        has_more = len(results) == PAGE_SIZE
        msg = f"Found {len(results)} images similar to cropped region."
        return (
            gallery_items, msg, PAGE_SIZE, gallery_items,
            new_metadata, query_emb.tolist(),
            gr.update(visible=has_more), selected_events,
            str(image_path), gr.Tabs(selected=1),
        )

    # ── Build UI ─────────────────────────────────────────────────────

    with gr.Blocks(title="PyCon JP Image Search", css=CUSTOM_CSS, head=CROP_TOOL_SCRIPT) as app:
        gr.Markdown("# PyCon JP Image Search")

        with gr.Tabs() as tabs:
            # ── Tab 1: Text Search ───────────────────────────────────
            with gr.TabItem("Text Search", id=0):
                with gr.Row():
                    text_input = gr.Textbox(
                        label="Search query",
                        placeholder="e.g. keynote speaker on stage",
                    )
                    text_event_filter = gr.Dropdown(
                        choices=event_names, value=[], multiselect=True,
                        label="Filter by Event",
                    )
                    text_btn = gr.Button("Search")

                text_preview_image = gr.Image(
                    label="Preview", visible=False, height=480,
                    elem_id="text-preview",
                )
                text_preview_caption = gr.Markdown("", visible=False)
                with gr.Row():
                    text_find_similar_btn = gr.Button("Find Similar", visible=False)
                    text_search_cropped_btn = gr.Button(
                        "Search Cropped", visible=False,
                        elem_id="text-search-cropped-btn",
                    )
                    text_copy_clipboard_btn = gr.Button(
                        "Copy to Clipboard", visible=False,
                        elem_id="text-copy-clipboard-btn",
                    )
                    text_close_btn = gr.Button("Close Preview", visible=False)
                text_thumb_strip = gr.Gallery(
                    label="", visible=False, rows=1, height=100,
                    allow_preview=False, elem_classes=["thumb-strip"],
                )

                text_gallery = gr.Gallery(
                    label="Results", columns=4, height="auto",
                    allow_preview=False, elem_classes=["full-height-gallery"],
                )
                text_info = gr.Markdown("")
                text_load_more_btn = gr.Button("Load More", visible=False)
                text_crop_data = gr.Textbox(visible=False, elem_id="text-crop-data")

                text_offset_state = gr.State(0)
                text_results_state = gr.State([])
                text_metadata_state = gr.State([])
                text_embedding_state = gr.State(None)
                text_selected_index_state = gr.State(None)

                def do_text_search(query: str, selected_events: list[str]) -> tuple:
                    if not query.strip():
                        return (
                            [], "Please enter a search query.",
                            0, [], [], None, gr.update(visible=False),
                        )
                    embedder = _get_embedder()
                    query_emb = embedder.embed_text(query)
                    results = search_images_by_text(
                        conn, query_embedding=query_emb,
                        model_name=SIGLIP_MODEL_NAME,
                        limit=PAGE_SIZE, offset=0,
                        event_names=selected_events or None,
                    )
                    gallery_items, metadata = _make_gallery_items(results)
                    has_more = len(results) == PAGE_SIZE
                    return (
                        gallery_items,
                        f"Found {len(results)} images for '{query}'.",
                        PAGE_SIZE, gallery_items, metadata,
                        query_emb.tolist(), gr.update(visible=has_more),
                    )

                def do_text_load_more(
                    selected_events: list[str], offset: int,
                    accumulated: list, accumulated_meta: list, query_emb_list,
                ) -> tuple:
                    if query_emb_list is None:
                        return (
                            accumulated, "No active search.",
                            offset, accumulated, accumulated_meta,
                            gr.update(visible=False),
                        )
                    query_emb = np.array(query_emb_list)
                    results = search_images_by_text(
                        conn, query_embedding=query_emb,
                        model_name=SIGLIP_MODEL_NAME,
                        limit=PAGE_SIZE, offset=offset,
                        event_names=selected_events or None,
                    )
                    new_items, new_meta = _make_gallery_items(results)
                    combined = accumulated + new_items
                    combined_meta = accumulated_meta + new_meta
                    has_more = len(results) == PAGE_SIZE
                    return (
                        combined,
                        f"Showing {len(combined)} images.",
                        offset + len(results), combined, combined_meta,
                        gr.update(visible=has_more),
                    )

                text_btn.click(
                    fn=do_text_search,
                    inputs=[text_input, text_event_filter],
                    outputs=[
                        text_gallery, text_info,
                        text_offset_state, text_results_state,
                        text_metadata_state, text_embedding_state,
                        text_load_more_btn,
                    ],
                ).then(
                    fn=_on_close_preview,
                    outputs=[
                        text_preview_image, text_preview_caption,
                        text_thumb_strip, text_close_btn,
                        text_find_similar_btn, text_search_cropped_btn,
                        text_copy_clipboard_btn,
                    ],
                )
                text_load_more_btn.click(
                    fn=do_text_load_more,
                    inputs=[
                        text_event_filter, text_offset_state,
                        text_results_state, text_metadata_state,
                        text_embedding_state,
                    ],
                    outputs=[
                        text_gallery, text_info,
                        text_offset_state, text_results_state,
                        text_metadata_state, text_load_more_btn,
                    ],
                )
                text_gallery.select(
                    fn=_on_gallery_select,
                    inputs=[text_results_state, text_metadata_state],
                    outputs=[
                        text_preview_image, text_preview_caption,
                        text_thumb_strip, text_close_btn,
                        text_find_similar_btn, text_search_cropped_btn,
                        text_copy_clipboard_btn, text_selected_index_state,
                    ],
                    js=SCROLL_AND_INIT_CROP_JS % ("text-preview", "text-preview"),
                )
                text_thumb_strip.select(
                    fn=_on_thumb_select,
                    inputs=[text_results_state, text_metadata_state],
                    outputs=[text_preview_image, text_preview_caption, text_selected_index_state],
                    js=REINIT_CROP_JS % "text-preview",
                )
                text_close_btn.click(
                    fn=_on_close_preview,
                    outputs=[
                        text_preview_image, text_preview_caption,
                        text_thumb_strip, text_close_btn,
                        text_find_similar_btn, text_search_cropped_btn,
                        text_copy_clipboard_btn,
                    ],
                )
                text_copy_clipboard_btn.click(
                    fn=None,
                    js=COPY_CLIPBOARD_JS % "text-preview",
                )

            # ── Tab 2: Image Search ──────────────────────────────────
            with gr.TabItem("Image Search", id=1):
                with gr.Row():
                    image_input = gr.Image(
                        label="Upload an image", type="filepath",
                    )
                    image_event_filter = gr.Dropdown(
                        choices=event_names, value=[], multiselect=True,
                        label="Filter by Event",
                    )
                    image_btn = gr.Button("Search Similar")

                img_preview_image = gr.Image(
                    label="Preview", visible=False, height=480,
                    elem_id="img-preview",
                )
                img_preview_caption = gr.Markdown("", visible=False)
                with gr.Row():
                    img_find_similar_btn = gr.Button("Find Similar", visible=False)
                    img_search_cropped_btn = gr.Button(
                        "Search Cropped", visible=False,
                        elem_id="img-search-cropped-btn",
                    )
                    img_copy_clipboard_btn = gr.Button(
                        "Copy to Clipboard", visible=False,
                        elem_id="img-copy-clipboard-btn",
                    )
                    img_close_btn = gr.Button("Close Preview", visible=False)
                img_thumb_strip = gr.Gallery(
                    label="", visible=False, rows=1, height=100,
                    allow_preview=False, elem_classes=["thumb-strip"],
                )

                image_gallery = gr.Gallery(
                    label="Results", columns=4, height="auto",
                    allow_preview=False, elem_classes=["full-height-gallery"],
                )
                image_info = gr.Markdown("")
                image_load_more_btn = gr.Button("Load More", visible=False)
                img_crop_data = gr.Textbox(visible=False, elem_id="img-crop-data")

                image_offset_state = gr.State(0)
                image_results_state = gr.State([])
                image_metadata_state = gr.State([])
                image_embedding_state = gr.State(None)
                image_selected_index_state = gr.State(None)

                def do_image_search(
                    image_path: str | None, selected_events: list[str],
                ) -> tuple:
                    if image_path is None:
                        return (
                            [], "Please upload an image.",
                            0, [], [], None, gr.update(visible=False),
                        )
                    embedder = _get_embedder()
                    query_emb = embedder.embed_images([Path(image_path)])
                    results = search_images_by_text(
                        conn, query_embedding=query_emb,
                        model_name=SIGLIP_MODEL_NAME,
                        limit=PAGE_SIZE, offset=0,
                        event_names=selected_events or None,
                    )
                    gallery_items, metadata = _make_gallery_items(results)
                    has_more = len(results) == PAGE_SIZE
                    return (
                        gallery_items,
                        f"Found {len(results)} similar images.",
                        PAGE_SIZE, gallery_items, metadata,
                        query_emb.tolist(), gr.update(visible=has_more),
                    )

                def do_image_load_more(
                    selected_events: list[str], offset: int,
                    accumulated: list, accumulated_meta: list, query_emb_list,
                ) -> tuple:
                    if query_emb_list is None:
                        return (
                            accumulated, "No active search.",
                            offset, accumulated, accumulated_meta,
                            gr.update(visible=False),
                        )
                    query_emb = np.array(query_emb_list)
                    results = search_images_by_text(
                        conn, query_embedding=query_emb,
                        model_name=SIGLIP_MODEL_NAME,
                        limit=PAGE_SIZE, offset=offset,
                        event_names=selected_events or None,
                    )
                    new_items, new_meta = _make_gallery_items(results)
                    combined = accumulated + new_items
                    combined_meta = accumulated_meta + new_meta
                    has_more = len(results) == PAGE_SIZE
                    return (
                        combined,
                        f"Showing {len(combined)} images.",
                        offset + len(results), combined, combined_meta,
                        gr.update(visible=has_more),
                    )

                image_btn.click(
                    fn=do_image_search,
                    inputs=[image_input, image_event_filter],
                    outputs=[
                        image_gallery, image_info,
                        image_offset_state, image_results_state,
                        image_metadata_state, image_embedding_state,
                        image_load_more_btn,
                    ],
                ).then(
                    fn=_on_close_preview,
                    outputs=[
                        img_preview_image, img_preview_caption,
                        img_thumb_strip, img_close_btn,
                        img_find_similar_btn, img_search_cropped_btn,
                        img_copy_clipboard_btn,
                    ],
                )
                image_load_more_btn.click(
                    fn=do_image_load_more,
                    inputs=[
                        image_event_filter, image_offset_state,
                        image_results_state, image_metadata_state,
                        image_embedding_state,
                    ],
                    outputs=[
                        image_gallery, image_info,
                        image_offset_state, image_results_state,
                        image_metadata_state, image_load_more_btn,
                    ],
                )
                image_gallery.select(
                    fn=_on_gallery_select,
                    inputs=[image_results_state, image_metadata_state],
                    outputs=[
                        img_preview_image, img_preview_caption,
                        img_thumb_strip, img_close_btn,
                        img_find_similar_btn, img_search_cropped_btn,
                        img_copy_clipboard_btn, image_selected_index_state,
                    ],
                    js=SCROLL_AND_INIT_CROP_JS % ("img-preview", "img-preview"),
                )
                img_thumb_strip.select(
                    fn=_on_thumb_select,
                    inputs=[image_results_state, image_metadata_state],
                    outputs=[img_preview_image, img_preview_caption, image_selected_index_state],
                    js=REINIT_CROP_JS % "img-preview",
                )
                img_close_btn.click(
                    fn=_on_close_preview,
                    outputs=[
                        img_preview_image, img_preview_caption,
                        img_thumb_strip, img_close_btn,
                        img_find_similar_btn, img_search_cropped_btn,
                        img_copy_clipboard_btn,
                    ],
                )
                img_copy_clipboard_btn.click(
                    fn=None,
                    js=COPY_CLIPBOARD_JS % "img-preview",
                )

        # ── Close previews on tab switch ─────────────────────────────
        def _on_tab_switch():
            close = _on_close_preview()
            return close + close  # both tabs

        tabs.select(
            fn=_on_tab_switch,
            outputs=[
                text_preview_image, text_preview_caption,
                text_thumb_strip, text_close_btn,
                text_find_similar_btn, text_search_cropped_btn,
                text_copy_clipboard_btn,
                img_preview_image, img_preview_caption,
                img_thumb_strip, img_close_btn,
                img_find_similar_btn, img_search_cropped_btn,
                img_copy_clipboard_btn,
            ],
        )

        # ── Cross-tab wiring (Find Similar / Search Cropped) ─────────
        _find_similar_outputs = [
            image_gallery, image_info,
            image_offset_state, image_results_state,
            image_metadata_state, image_embedding_state,
            image_load_more_btn, image_event_filter,
            image_input, tabs,
        ]

        _text_close_outputs = [
            text_preview_image, text_preview_caption,
            text_thumb_strip, text_close_btn,
            text_find_similar_btn, text_search_cropped_btn,
            text_copy_clipboard_btn,
        ]
        _img_close_outputs = [
            img_preview_image, img_preview_caption,
            img_thumb_strip, img_close_btn,
            img_find_similar_btn, img_search_cropped_btn,
            img_copy_clipboard_btn,
        ]

        text_find_similar_btn.click(
            fn=_do_find_similar,
            inputs=[text_selected_index_state, text_metadata_state, text_event_filter],
            outputs=_find_similar_outputs,
        ).then(
            fn=_on_close_preview,
            outputs=_text_close_outputs,
        )
        img_find_similar_btn.click(
            fn=_do_find_similar,
            inputs=[image_selected_index_state, image_metadata_state, image_event_filter],
            outputs=_find_similar_outputs,
        ).then(
            fn=_on_close_preview,
            outputs=_img_close_outputs,
        )

        # Search Cropped: JS extracts crop rect → JSON → Python fetches & crops
        text_search_cropped_btn.click(
            fn=None,
            js=CROP_TO_JSON_JS % "text-preview",
            outputs=[text_crop_data],
        ).then(
            fn=_do_search_cropped,
            inputs=[text_crop_data, text_event_filter],
            outputs=_find_similar_outputs,
        ).then(
            fn=_on_close_preview,
            outputs=_text_close_outputs,
        )
        img_search_cropped_btn.click(
            fn=None,
            js=CROP_TO_JSON_JS % "img-preview",
            outputs=[img_crop_data],
        ).then(
            fn=_do_search_cropped,
            inputs=[img_crop_data, image_event_filter],
            outputs=_find_similar_outputs,
        ).then(
            fn=_on_close_preview,
            outputs=_img_close_outputs,
        )

    return app
