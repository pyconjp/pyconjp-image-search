"""Image search UI with Gradio."""


def main() -> None:
    """CLI entry point for Gradio search UI."""
    from pyconjp_image_search.config import DATA_DIR
    from pyconjp_image_search.search.app import create_app

    app = create_app()
    app.launch(allowed_paths=[str(DATA_DIR)])
