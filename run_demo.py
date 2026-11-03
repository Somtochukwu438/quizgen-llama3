"""Launch the Gradio quiz demo locally (requires an NVIDIA GPU with CUDA).

Run from the repository root:  python run_demo.py          (local link)
                               python run_demo.py --share  (public gradio.live link)
"""
import argparse

from app.app import build_app
from src.generate import load_model
from src.notes import load_notes
from src.rag import NotesIndex


def main():
    parser = argparse.ArgumentParser(description="Launch the quiz generator demo.")
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = parser.parse_args()

    model, tok = load_model("adapters")
    index = NotesIndex(load_notes("data/notes"))
    build_app(model, tok, index).launch(share=args.share)


if __name__ == "__main__":
    main()