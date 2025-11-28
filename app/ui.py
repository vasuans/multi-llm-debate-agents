"""
ui.py

Defines the Gradio interface that calls the debate pipeline.

- Supports live updates (step-by-step debate).
- Allows choosing which models are Debater A, Debater B, and Judge.
- Uses HTML for the transcript so colored blocks render correctly.
"""

import gradio as gr

from .state import DebateState
from .nodes import (
    node_load_memory,
    node_opening,
    node_rebuttal_round_1,
    node_rebuttal_round_2,
    node_judge,
    node_store_memory,
    node_assemble,
)
from .config import DEFAULT_TEMPERATURE, MIN_TEMPERATURE, MAX_TEMPERATURE


def _render_outputs(state: DebateState, show_final: bool) -> tuple[str, str]:
    """
    Build the final + transcript HTML from current state.
    If show_final is False, we show a placeholder in the final answer area.
    """

    transcript_html = "\n\n".join(state.get("transcript_sections", []))

    if show_final:
        winner = state.get("winner", "Unknown")
        final_answer = state.get("final_answer", "").strip()
        if final_answer:
            final_md = f"### 🏁 Final Answer (Winner: {winner})\n\n{final_answer}"
        else:
            final_md = "Final answer not available."
    else:
        final_md = "🧠 Debate in progress... please wait for the final answer."

    return final_md, transcript_html


def debate_live(
    question: str,
    creativity: float,
    debater_a_model: str,
    debater_b_model: str,
    judge_model: str,
):
    """
    Live debate runner for Gradio.

    This is a generator: it yields multiple times so the UI updates
    after each phase (memory, opening, rebuttals, judge).
    """

    question = (question or "").strip()
    if not question:
        yield "Please enter a question or topic.", ""
        return

    # Initialize state
    state: DebateState = {
        "question": question,
        "temperature": creativity,
        "transcript_sections": [],
        "memory_snippets": [],
        "debater_a_model": debater_a_model,
        "debater_b_model": debater_b_model,
        "judge_model": judge_model,
    }

    # 1) Load memory
    state = node_load_memory(state)
    yield _render_outputs(state, show_final=False)

    # 2) Opening statements
    state = node_opening(state)
    yield _render_outputs(state, show_final=False)

    # 3) Rebuttal round 1
    state = node_rebuttal_round_1(state)
    yield _render_outputs(state, show_final=False)

    # 4) Rebuttal round 2
    state = node_rebuttal_round_2(state)
    yield _render_outputs(state, show_final=False)

    # 5) Judge
    state = node_judge(state)
    yield _render_outputs(state, show_final=True)

    # 6) Store memory + assemble transcript
    state = node_store_memory(state)
    state = node_assemble(state)
    yield _render_outputs(state, show_final=True)


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

            **Multi-LLM debate with memory (ChromaDB) + live updates**

            - Debater A: configurable (OpenAI / Grok)  
            - Debater B: configurable (OpenAI / Grok)  
            - Judge: configurable (Gemini / OpenAI)  
            - Memory: ChromaDB (stores past debates and recalls similar ones)
            """
        )

        with gr.Row():
            # Left: controls (narrower)
            with gr.Column(scale=1, min_width=320):
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

                debater_a_dropdown = gr.Dropdown(
                    label="Debater A Model",
                    choices=[
                        ("OpenAI (gpt-4.1-mini)", "openai"),
                        ("Grok (grok-3-mini)", "grok"),
                    ],
                    value="openai",
                )

                debater_b_dropdown = gr.Dropdown(
                    label="Debater B Model",
                    choices=[
                        ("OpenAI (gpt-4.1-mini)", "openai"),
                        ("Grok (grok-3-mini)", "grok"),
                    ],
                    value="grok",
                )

                judge_dropdown = gr.Dropdown(
                    label="Judge Model",
                    choices=[
                        ("Gemini (gemini-2.0-flash-lite)", "gemini"),
                        ("OpenAI (gpt-4.1-mini)", "openai"),
                    ],
                    value="gemini",
                )

                run_button = gr.Button("🔥 Run 2-Round Debate (Live)", variant="primary")

                gr.Markdown(
                    """
                    ### 🎥 Screen Recording Tips

                    - Ask a few related questions in a row to show MEMORY effect.
                    - Change which models are Debater A / Debater B / Judge between runs.
                    - Example sequence:
                      1. *LangChain vs LangGraph for beginners?*
                      2. *Given earlier debates, which is better for multi-agent workflows?*
                      3. *How should I structure my first LangGraph project?*
                    """
                )

            # Right: outputs (wider)
            with gr.Column(scale=2):
                final_output = gr.Markdown(
                    label="Final Answer",
                    value="Final answer will appear here after the debate.",
                )

                gr.Markdown("### 📜 Live Debate Transcript")
                transcript_output = gr.HTML(
                    value="<p>The live 2-round debate + memory context will appear here.</p>",
                )

        # Wire button to live debate generator
        run_button.click(
            fn=debate_live,
            inputs=[
                question_input,
                creativity_slider,
                debater_a_dropdown,
                debater_b_dropdown,
                judge_dropdown,
            ],
            outputs=[final_output, transcript_output],
        )

    return demo
