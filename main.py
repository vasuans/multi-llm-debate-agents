"""
main.py

Entry point. Launches the Gradio UI.
"""

from app.ui import create_ui


def main() -> None:
    demo = create_ui()
    demo.launch()


if __name__ == "__main__":
    main()
