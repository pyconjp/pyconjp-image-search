"""Gradio application for image search UI."""

import numpy as np
import gradio as gr

from pyconjp_image_search.config import DATA_DIR, SIGLIP_MODEL_NAME
from pyconjp_image_search.db import get_connection
from pyconjp_image_search.models import ImageMetadata
from pyconjp_image_search.search.query import (
    get_event_names,
    search_images_by_text,
)

PAGE_SIZE = 20

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
"""

SCROLL_TO_JS = """
(args) => {
    setTimeout(() => {
        const el = document.getElementById('%s');
        if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});
    }, 100);
}
"""


def create_app() -> gr.Blocks:
    """Create and return the Gradio Blocks app."""
    conn = get_connection()

    event_names = get_event_names(conn)

    # Lazy-loaded embedder for text search
    _embedder_cache: dict = {}

    def _get_embedder():
        if "instance" not in _embedder_cache:
            from pyconjp_image_search.embedding.siglip import SigLIPEmbedder

            _embedder_cache["instance"] = SigLIPEmbedder(model_name=SIGLIP_MODEL_NAME)
        return _embedder_cache["instance"]

    def _make_gallery_items(
        results: list[tuple[ImageMetadata, float]],
    ) -> list[tuple[str, str]]:
        items = []
        for meta, score in results:
            path = str(DATA_DIR / meta.relative_path) if meta.relative_path else meta.image_url
            caption = f"score: {score:.3f} | {meta.event_name}"
            items.append((path, caption))
        return items

    def _on_gallery_select(gallery_items: list, evt: gr.EventData):
        """Show clicked image in the preview area with thumbnail strip."""
        index = evt._data.get("index")
        if index is not None and gallery_items:
            item = gallery_items[index]
            image_path = item[0] if isinstance(item, (list, tuple)) else item
            caption = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else ""
            return (
                gr.update(value=image_path, visible=True),
                gr.update(value=caption, visible=True),
                gr.update(value=gallery_items, visible=True),
                gr.update(visible=True),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def _on_thumb_select(gallery_items: list, evt: gr.EventData):
        """Update preview when a thumbnail is clicked."""
        index = evt._data.get("index")
        if index is not None and gallery_items:
            item = gallery_items[index]
            image_path = item[0] if isinstance(item, (list, tuple)) else item
            caption = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else ""
            return gr.update(value=image_path), gr.update(value=caption)
        return gr.update(), gr.update()

    def _on_close_preview():
        return (
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
            gr.update(value=None, visible=False),
            gr.update(visible=False),
        )

    with gr.Blocks(title="PyCon JP Image Search", css=CUSTOM_CSS) as app:
        gr.Markdown("# PyCon JP Image Search")

        with gr.Tabs():
            # Tab 1: Text Search
            with gr.TabItem("Text Search"):
                with gr.Row():
                    text_input = gr.Textbox(
                        label="Search query",
                        placeholder="e.g. keynote speaker on stage",
                    )
                    text_event_filter = gr.Dropdown(
                        choices=event_names,
                        value=[],
                        multiselect=True,
                        label="Filter by Event",
                    )
                    text_btn = gr.Button("Search")

                # Preview area (hidden by default)
                text_preview_image = gr.Image(
                    label="Preview", visible=False, height=480,
                    elem_id="text-preview",
                )
                text_preview_caption = gr.Markdown("", visible=False)
                text_thumb_strip = gr.Gallery(
                    label="", visible=False, rows=1, height=100,
                    allow_preview=False,
                    elem_classes=["thumb-strip"],
                )
                text_close_btn = gr.Button("Close Preview", visible=False)

                text_gallery = gr.Gallery(
                    label="Results", columns=4, height="auto",
                    allow_preview=False,
                    elem_classes=["full-height-gallery"],
                )
                text_info = gr.Markdown("")
                text_load_more_btn = gr.Button("Load More", visible=False)

                # State for pagination
                text_offset_state = gr.State(0)
                text_results_state = gr.State([])
                text_embedding_state = gr.State(None)

                def do_text_search(
                    query: str, selected_events: list[str],
                ) -> tuple:
                    if not query.strip():
                        return (
                            [], "Please enter a search query.",
                            0, [], None, gr.update(visible=False),
                        )
                    embedder = _get_embedder()
                    query_emb = embedder.embed_text(query)
                    results = search_images_by_text(
                        conn,
                        query_embedding=query_emb,
                        model_name=SIGLIP_MODEL_NAME,
                        limit=PAGE_SIZE,
                        offset=0,
                        event_names=selected_events or None,
                    )
                    gallery_items = _make_gallery_items(results)
                    has_more = len(results) == PAGE_SIZE
                    msg = f"Found {len(results)} images for '{query}'."
                    return (
                        gallery_items,
                        msg,
                        PAGE_SIZE,
                        gallery_items,
                        query_emb.tolist(),
                        gr.update(visible=has_more),
                    )

                def do_text_load_more(
                    selected_events: list[str],
                    offset: int,
                    accumulated: list,
                    query_emb_list,
                ) -> tuple:
                    if query_emb_list is None:
                        return (
                            accumulated, "No active search.",
                            offset, accumulated, gr.update(visible=False),
                        )
                    query_emb = np.array(query_emb_list)
                    results = search_images_by_text(
                        conn,
                        query_embedding=query_emb,
                        model_name=SIGLIP_MODEL_NAME,
                        limit=PAGE_SIZE,
                        offset=offset,
                        event_names=selected_events or None,
                    )
                    new_items = _make_gallery_items(results)
                    combined = accumulated + new_items
                    has_more = len(results) == PAGE_SIZE
                    new_offset = offset + len(results)
                    msg = f"Showing {len(combined)} images."
                    return (
                        combined,
                        msg,
                        new_offset,
                        combined,
                        gr.update(visible=has_more),
                    )

                text_btn.click(
                    fn=do_text_search,
                    inputs=[text_input, text_event_filter],
                    outputs=[
                        text_gallery, text_info,
                        text_offset_state, text_results_state,
                        text_embedding_state, text_load_more_btn,
                    ],
                )
                text_load_more_btn.click(
                    fn=do_text_load_more,
                    inputs=[
                        text_event_filter, text_offset_state,
                        text_results_state, text_embedding_state,
                    ],
                    outputs=[
                        text_gallery, text_info,
                        text_offset_state, text_results_state,
                        text_load_more_btn,
                    ],
                )
                text_gallery.select(
                    fn=_on_gallery_select,
                    inputs=[text_results_state],
                    outputs=[text_preview_image, text_preview_caption, text_thumb_strip, text_close_btn],
                    js=SCROLL_TO_JS % "text-preview",
                )
                text_thumb_strip.select(
                    fn=_on_thumb_select,
                    inputs=[text_results_state],
                    outputs=[text_preview_image, text_preview_caption],
                )
                text_close_btn.click(
                    fn=_on_close_preview,
                    outputs=[text_preview_image, text_preview_caption, text_thumb_strip, text_close_btn],
                )

            # Tab 2: Image Search
            with gr.TabItem("Image Search"):
                with gr.Row():
                    image_input = gr.Image(
                        label="Upload an image",
                        type="filepath",
                    )
                    image_event_filter = gr.Dropdown(
                        choices=event_names,
                        value=[],
                        multiselect=True,
                        label="Filter by Event",
                    )
                    image_btn = gr.Button("Search Similar")

                # Preview area (hidden by default)
                img_preview_image = gr.Image(
                    label="Preview", visible=False, height=480,
                    elem_id="img-preview",
                )
                img_preview_caption = gr.Markdown("", visible=False)
                img_thumb_strip = gr.Gallery(
                    label="", visible=False, rows=1, height=100,
                    allow_preview=False,
                    elem_classes=["thumb-strip"],
                )
                img_close_btn = gr.Button("Close Preview", visible=False)

                image_gallery = gr.Gallery(
                    label="Results", columns=4, height="auto",
                    allow_preview=False,
                    elem_classes=["full-height-gallery"],
                )
                image_info = gr.Markdown("")
                image_load_more_btn = gr.Button("Load More", visible=False)

                # State for pagination
                image_offset_state = gr.State(0)
                image_results_state = gr.State([])
                image_embedding_state = gr.State(None)

                def do_image_search(
                    image_path: str | None, selected_events: list[str],
                ) -> tuple:
                    if image_path is None:
                        return (
                            [], "Please upload an image.",
                            0, [], None, gr.update(visible=False),
                        )
                    from pathlib import Path

                    embedder = _get_embedder()
                    query_emb = embedder.embed_images([Path(image_path)])
                    results = search_images_by_text(
                        conn,
                        query_embedding=query_emb,
                        model_name=SIGLIP_MODEL_NAME,
                        limit=PAGE_SIZE,
                        offset=0,
                        event_names=selected_events or None,
                    )
                    gallery_items = _make_gallery_items(results)
                    has_more = len(results) == PAGE_SIZE
                    msg = f"Found {len(results)} similar images."
                    return (
                        gallery_items,
                        msg,
                        PAGE_SIZE,
                        gallery_items,
                        query_emb.tolist(),
                        gr.update(visible=has_more),
                    )

                def do_image_load_more(
                    selected_events: list[str],
                    offset: int,
                    accumulated: list,
                    query_emb_list,
                ) -> tuple:
                    if query_emb_list is None:
                        return (
                            accumulated, "No active search.",
                            offset, accumulated, gr.update(visible=False),
                        )
                    query_emb = np.array(query_emb_list)
                    results = search_images_by_text(
                        conn,
                        query_embedding=query_emb,
                        model_name=SIGLIP_MODEL_NAME,
                        limit=PAGE_SIZE,
                        offset=offset,
                        event_names=selected_events or None,
                    )
                    new_items = _make_gallery_items(results)
                    combined = accumulated + new_items
                    has_more = len(results) == PAGE_SIZE
                    new_offset = offset + len(results)
                    msg = f"Showing {len(combined)} images."
                    return (
                        combined,
                        msg,
                        new_offset,
                        combined,
                        gr.update(visible=has_more),
                    )

                image_btn.click(
                    fn=do_image_search,
                    inputs=[image_input, image_event_filter],
                    outputs=[
                        image_gallery, image_info,
                        image_offset_state, image_results_state,
                        image_embedding_state, image_load_more_btn,
                    ],
                )
                image_load_more_btn.click(
                    fn=do_image_load_more,
                    inputs=[
                        image_event_filter, image_offset_state,
                        image_results_state, image_embedding_state,
                    ],
                    outputs=[
                        image_gallery, image_info,
                        image_offset_state, image_results_state,
                        image_load_more_btn,
                    ],
                )
                image_gallery.select(
                    fn=_on_gallery_select,
                    inputs=[image_results_state],
                    outputs=[img_preview_image, img_preview_caption, img_thumb_strip, img_close_btn],
                    js=SCROLL_TO_JS % "img-preview",
                )
                img_thumb_strip.select(
                    fn=_on_thumb_select,
                    inputs=[image_results_state],
                    outputs=[img_preview_image, img_preview_caption],
                )
                img_close_btn.click(
                    fn=_on_close_preview,
                    outputs=[img_preview_image, img_preview_caption, img_thumb_strip, img_close_btn],
                )

    return app
