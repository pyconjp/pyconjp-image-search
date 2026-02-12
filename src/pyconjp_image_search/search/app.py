"""Gradio application for image search UI."""

import gradio as gr

from pyconjp_image_search.config import DATA_DIR, SIGLIP_MODEL_NAME
from pyconjp_image_search.db import get_connection
from pyconjp_image_search.search.query import (
    get_event_names,
    get_event_years,
    search_images,
    search_images_by_text,
)


def create_app() -> gr.Blocks:
    """Create and return the Gradio Blocks app."""
    conn = get_connection()

    event_names = get_event_names(conn)
    event_years = get_event_years(conn)

    # Lazy-loaded embedder for text search
    _embedder_cache: dict = {}

    def _get_embedder():
        if "instance" not in _embedder_cache:
            from pyconjp_image_search.embedding.siglip import SigLIPEmbedder

            _embedder_cache["instance"] = SigLIPEmbedder(model_name=SIGLIP_MODEL_NAME)
        return _embedder_cache["instance"]

    with gr.Blocks(title="PyCon JP Image Search") as app:
        gr.Markdown("# PyCon JP Image Search")

        with gr.Tabs():
            # Tab 1: Browse
            with gr.TabItem("Browse"):
                with gr.Row():
                    event_name_dd = gr.Dropdown(
                        choices=[""] + event_names,
                        value="",
                        label="Event Name",
                    )
                    event_year_dd = gr.Dropdown(
                        choices=[""] + [str(y) for y in event_years],
                        value="",
                        label="Event Year",
                    )
                    browse_btn = gr.Button("Search")

                browse_gallery = gr.Gallery(label="Results", columns=4, height="auto")
                browse_info = gr.Markdown("")

                def do_browse(event_name: str, event_year: str) -> tuple:
                    results = search_images(
                        conn,
                        event_name=event_name or None,
                        event_year=int(event_year) if event_year else None,
                    )
                    images = []
                    for img in results:
                        if img.relative_path:
                            images.append(str(DATA_DIR / img.relative_path))
                        else:
                            images.append(img.image_url)
                    msg = f"Found {len(results)} images."
                    return images, msg

                browse_btn.click(
                    fn=do_browse,
                    inputs=[event_name_dd, event_year_dd],
                    outputs=[browse_gallery, browse_info],
                )

            # Tab 2: Text Search
            with gr.TabItem("Text Search"):
                with gr.Row():
                    text_input = gr.Textbox(
                        label="Search query",
                        placeholder="e.g. keynote speaker on stage",
                    )
                    text_btn = gr.Button("Search")

                text_gallery = gr.Gallery(label="Results", columns=4, height="auto")
                text_info = gr.Markdown("")

                def do_text_search(query: str) -> tuple:
                    if not query.strip():
                        return [], "Please enter a search query."
                    embedder = _get_embedder()
                    query_emb = embedder.embed_text(query)
                    results = search_images_by_text(
                        conn,
                        query_embedding=query_emb,
                        model_name=SIGLIP_MODEL_NAME,
                        limit=20,
                    )
                    images = []
                    captions = []
                    for meta, score in results:
                        if meta.relative_path:
                            images.append(str(DATA_DIR / meta.relative_path))
                        else:
                            images.append(meta.image_url)
                        captions.append(
                            f"score: {score:.3f} | {meta.event_name} {meta.event_year}"
                        )
                    gallery_items = list(zip(images, captions))
                    msg = f"Found {len(results)} images for '{query}'."
                    return gallery_items, msg

                text_btn.click(
                    fn=do_text_search,
                    inputs=[text_input],
                    outputs=[text_gallery, text_info],
                )

            # Tab 3: Image Search
            with gr.TabItem("Image Search"):
                with gr.Row():
                    image_input = gr.Image(
                        label="Upload an image",
                        type="filepath",
                    )
                    image_btn = gr.Button("Search Similar")

                image_gallery = gr.Gallery(label="Results", columns=4, height="auto")
                image_info = gr.Markdown("")

                def do_image_search(image_path: str | None) -> tuple:
                    if image_path is None:
                        return [], "Please upload an image."
                    from pathlib import Path

                    embedder = _get_embedder()
                    query_emb = embedder.embed_images([Path(image_path)])
                    results = search_images_by_text(
                        conn,
                        query_embedding=query_emb,
                        model_name=SIGLIP_MODEL_NAME,
                        limit=20,
                    )
                    images = []
                    captions = []
                    for meta, score in results:
                        if meta.relative_path:
                            images.append(str(DATA_DIR / meta.relative_path))
                        else:
                            images.append(meta.image_url)
                        captions.append(
                            f"score: {score:.3f} | {meta.event_name} {meta.event_year}"
                        )
                    gallery_items = list(zip(images, captions))
                    msg = f"Found {len(results)} similar images."
                    return gallery_items, msg

                image_btn.click(
                    fn=do_image_search,
                    inputs=[image_input],
                    outputs=[image_gallery, image_info],
                )

    return app
