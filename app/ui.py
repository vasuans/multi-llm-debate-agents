"""
ui.py

Defines the Gradio interface that calls run_debate().
"""

import gradio as gr

from .graph_runner import run_debate
from .config import DEFAULT_TEMPERATURE, MIN_TEMPERATURE, MAX_TEMPERATURE


def create_ui() -> gr.Blocks:
    """
    Build and return the Gradio Blocks app.
    """

    with gr.Blocks(
        title="LLM Debate Arena (LangGraph + OpenAI + Grok + Gemini + Memory)"
    ) as demo:
        gr.Markdown(
            """
            # 🤖 LLM Debate Arena (LangGraph + Memory)
            **Multi-LLM debate with memory (ChromaDB)**

            - Debater A: OpenAI `gpt-4.1-mini`  
            - Debater B: Grok `grok-3-mini`  
            - Judge: Gemini `gemini-2.0-flash-lite`  
            - Memory: ChromaDB (stores past debates and recalls similar ones)
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                question_input = gr.Textbox(
                    label="Question / Topic",
                    placeholder="Example: Should I start my first AI project with LangChain or LangGraph?",
                    lines=4,
                )

                creativity_slider = gr.Slider(
                    minimum=MIN_TEMPERATURE,
                    maximum=MAX_TEMPERATURE,
                    value=DEFAULT_TEMPERATURE,
                    step=0.1,
                    label="Creativity (temperature)",
                    info="Higher = more creative arguments, lower = more focused.",
                )

                run_button = gr.Button("🔥 Run 2-Round Debate", variant="primary")

                gr.Markdown(
                    """
                    ### 🎥 Screen Recording Tips

                    - Ask a few related questions in a row to show MEMORY effect.
                    - Example sequence:
                      1. *LangChain vs LangGraph for beginners?*
                      2. *Given earlier debates, which is better for multi-agent workflows?*
                      3. *How should I structure my first LangGraph project?*
                    """
                )

            with gr.Column(scale=1):
                final_output = gr.Markdown(
                    label="Final Answer",
                    value="Final answer will appear here after the debate.",
                )

                with gr.Tab("📜 Full Debate Transcript"):
                    transcript_output = gr.Markdown(
                        value="The entire 2-round debate + memory context will appear here."
                    )

        # Wire button to run_debate()
        run_button.click(
            fn=run_debate,
            inputs=[question_input, creativity_slider],
            outputs=[final_output, transcript_output],
        )

    return demo
