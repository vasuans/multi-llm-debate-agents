"""
nodes.py

LangGraph node functions for the debate:
- load_memory
- opening
- rebuttal_round_1
- rebuttal_round_2
- judge
- store_memory
- assemble

Transcript is built as HTML fragments in transcript_sections.
"""

from typing import List

from .state import DebateState
from .clients import (
    call_openai_debater,
    call_grok_debater,
    call_gemini_judge,
)
from .memory import load_relevant_memories, store_debate_memory

# Colors (used for borders and labels)
DEBATER_A_COLOR = "#1976d2"  # blue
DEBATER_B_COLOR = "#2e7d32"  # green
JUDGE_COLOR = "#424242"      # dark grey


def _html_block(border_color: str, title: str, body: str) -> str:
    """
    White-ish box with colored left border.
    Force dark text so it stays readable in dark theme.
    """
    safe_body = body.replace("\n", "<br>")
    return (
        '<div style="'
        'background-color:#fdfdfd;'
        f' border-left:4px solid {border_color};'
        ' padding:10px 12px;'
        ' border-radius:6px;'
        ' margin-bottom:10px;'
        ' color:#000000 !important;'
        ' line-height:1.6;'
        ' font-size:15px;'
        '">'
        f'<div style="font-weight:650; margin-bottom:6px;">{title}</div>'
        f'<div>{safe_body}</div>'
        '</div>'
    )



def _model_human_name(model_key: str) -> str:
    """
    Map internal model keys to human-readable names.
    """
    if model_key == "grok":
        return "Grok (grok-3-mini)"
    if model_key == "openai":
        return "OpenAI (gpt-4.1-mini)"
    return f"Unknown model ({model_key})"


def _opening_title(model_key: str) -> str:
    """
    Title for opening arguments: 'OpenAI (gpt-4.1-mini) – Opening'
    """
    return f"{_model_human_name(model_key)} – Opening"


def _rebuttal_title(model_key: str, round_num: int) -> str:
    """
    Title for rebuttal blocks: 'Grok (grok-3-mini) – Rebuttal Round 2'
    """
    return f"{_model_human_name(model_key)} – Rebuttal Round {round_num}"


# -------------------------------
# Memory nodes
# -------------------------------

def node_load_memory(state: DebateState) -> DebateState:
    """
    Load relevant past debates into memory_snippets.
    Do NOT show them in UI.
    """
    question = state["question"]
    snippets = load_relevant_memories(question, top_k=3)

    state["memory_snippets"] = snippets

    # Ensure transcript_sections exists
    if "transcript_sections" not in state or state["transcript_sections"] is None:
        state["transcript_sections"] = []

    return state


def node_store_memory(state: DebateState) -> DebateState:
    """
    Store final debate result into Chroma.
    """
    question = state.get("question", "")
    final_answer = state.get("final_answer", "")
    winner = state.get("winner", "Unknown")

    store_debate_memory(question, final_answer, winner)
    return state


# -------------------------------
# Helper functions for debaters
# -------------------------------

def _call_debater(
    model_key: str,
    messages: List[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    """
    Route to correct LLM:
    - 'openai' -> OpenAI
    - 'grok'   -> Grok (xAI)
    """
    if model_key == "grok":
        return call_grok_debater(messages, temperature=temperature, max_tokens=max_tokens)
    return call_openai_debater(messages, temperature=temperature, max_tokens=max_tokens)


def _short_rebuttal_for_a(
    question: str,
    other_text: str,
    temperature: float,
    model_key: str,
) -> str:
    """
    Debater A rebuts B in short bullet points.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are Debater A. Briefly rebut Debater B.\n"
                "Rules:\n"
                "- Max 3 bullet points.\n"
                "- Each bullet under 20 words.\n"
                "- Be precise, not rude."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nDebater B's latest answer:\n{other_text}",
        },
    ]
    return _call_debater(model_key, messages, temperature=temperature, max_tokens=120)


def _short_rebuttal_for_b(
    question: str,
    other_text: str,
    temperature: float,
    model_key: str,
) -> str:
    """
    Debater B rebuts A.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are Debater B. Briefly rebut Debater A.\n"
                "Rules:\n"
                "- Max 3 bullet points.\n"
                "- Each bullet under 20 words.\n"
                "- Be precise, not rude."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nDebater A's latest answer:\n{other_text}",
        },
    ]
    return _call_debater(model_key, messages, temperature=temperature, max_tokens=120)


# -------------------------------
# Main node functions
# -------------------------------

def node_opening(state: DebateState) -> DebateState:
    """
    Both debaters give a short opening answer.
    Memory snippets are used only as extra context, not shown.
    """

    question = state["question"]
    temperature = state["temperature"]
    model_a = state["debater_a_model"]
    model_b = state["debater_b_model"]
    memory_snippets: List[str] = state.get("memory_snippets", [])

    memory_context = ""
    if memory_snippets:
        memory_context = (
            "Here are some relevant past debates. You may reuse useful ideas, "
            "but do not copy sentences word-for-word:\n\n"
            + "\n\n---\n\n".join(memory_snippets)
        )
        memory_context = memory_context[:1200]

    # Debater A – opening
    sys_a = (
        "You are Debater A. Answer the user's question briefly.\n"
        "Rules:\n"
        "- Max ~120 words.\n"
        "- Use at most 3 bullet points OR 2 short paragraphs.\n"
    )
    if memory_context:
        sys_a += "\nYou also have access to some memory:\n" + memory_context

    messages_a = [
        {"role": "system", "content": sys_a},
        {"role": "user", "content": f"Question: {question}"},
    ]
    opening_a = _call_debater(
        model_a,
        messages_a,
        temperature=temperature,
        max_tokens=220,
    )

    # Debater B – opening
    sys_b = (
        "You are Debater B. Answer the user's question briefly.\n"
        "Rules:\n"
        "- Max ~120 words.\n"
        "- Use at most 3 bullet points OR 2 short paragraphs.\n"
        "- Focus slightly more on practical examples.\n"
    )
    if memory_context:
        sys_b += "\nYou also have access to some memory:\n" + memory_context

    messages_b = [
        {"role": "system", "content": sys_b},
        {"role": "user", "content": f"Question: {question}"},
    ]
    opening_b = _call_debater(
        model_b,
        messages_b,
        temperature=temperature,
        max_tokens=220,
    )

    sections = state.get("transcript_sections", [])
    sections.append("<h2>🧠 Question</h2>")
    sections.append(f"<p>{question}</p>")
    sections.append("<h2>🎙️ Opening Statements</h2>")

    sections.append(_html_block(DEBATER_A_COLOR, _opening_title(model_a), opening_a))
    sections.append(_html_block(DEBATER_B_COLOR, _opening_title(model_b), opening_b))

    state["opening_a"] = opening_a
    state["opening_b"] = opening_b
    state["rebuttals_a"] = []
    state["rebuttals_b"] = []
    state["transcript_sections"] = sections

    return state


def node_rebuttal_round_1(state: DebateState) -> DebateState:
    """
    First short rebuttal round.
    """

    question = state["question"]
    temp = state["temperature"]
    model_a = state["debater_a_model"]
    model_b = state["debater_b_model"]

    opening_a = state["opening_a"]
    opening_b = state["opening_b"]

    rebuttal_a = _short_rebuttal_for_a(question, opening_b, temp, model_a)
    rebuttal_b = _short_rebuttal_for_b(question, opening_a, temp, model_b)

    state["rebuttals_a"].append(rebuttal_a)
    state["rebuttals_b"].append(rebuttal_b)

    sections = state["transcript_sections"]
    sections.append("<h2>🔁 Rebuttal Round 1</h2>")

    sections.append(_html_block(DEBATER_A_COLOR, _rebuttal_title(model_a, 1), rebuttal_a))
    sections.append(_html_block(DEBATER_B_COLOR, _rebuttal_title(model_b, 1), rebuttal_b))
    state["transcript_sections"] = sections

    return state


def node_rebuttal_round_2(state: DebateState) -> DebateState:
    """
    Second short rebuttal round.
    """

    question = state["question"]
    temp = state["temperature"]
    model_a = state["debater_a_model"]
    model_b = state["debater_b_model"]

    latest_a = state["rebuttals_a"][-1]
    latest_b = state["rebuttals_b"][-1]

    rebuttal_a2 = _short_rebuttal_for_a(question, latest_b, temp, model_a)
    rebuttal_b2 = _short_rebuttal_for_b(question, latest_a, temp, model_b)

    state["rebuttals_a"].append(rebuttal_a2)
    state["rebuttals_b"].append(rebuttal_b2)

    sections = state["transcript_sections"]
    sections.append("<h2>🔁 Rebuttal Round 2</h2>")

    sections.append(_html_block(DEBATER_A_COLOR, _rebuttal_title(model_a, 2), rebuttal_a2))
    sections.append(_html_block(DEBATER_B_COLOR, _rebuttal_title(model_b, 2), rebuttal_b2))
    state["transcript_sections"] = sections

    return state


def node_judge(state: DebateState) -> DebateState:
    """
    Judge (Gemini or OpenAI) picks a winner and synthesizes a final answer.

    Winner / Reason / Final are parsed and rendered as
    separate colored lines inside the judge block.
    """

    question = state["question"]
    opening_a = state["opening_a"]
    opening_b = state["opening_b"]
    latest_a = state["rebuttals_a"][-1]
    latest_b = state["rebuttals_b"][-1]

    model_a = state["debater_a_model"]
    model_b = state["debater_b_model"]
    judge_model = state["judge_model"]

    judge_instructions = (
        "You are the Judge of a debate between two AI debaters (A and B).\n"
        "Your job:\n"
        "1) Decide who argued better overall (A, B, or tie).\n"
        "2) Briefly explain why.\n"
        "3) Give a final concise answer for the user.\n\n"
        "Output format (plain text, short):\n"
        "Winner: A / B / tie\n"
        "Reason: <1–3 short sentences>\n"
        "Final: <short final answer, max ~80 words>\n"
    )

    debate_context = (
        f"Question:\n{question}\n\n"
        f"Debater A (Opening):\n{opening_a}\n\n"
        f"Debater B (Opening):\n{opening_b}\n\n"
        f"Debater A (Latest Rebuttal):\n{latest_a}\n\n"
        f"Debater B (Latest Rebuttal):\n{latest_b}\n"
    )

    # Decide judge LLM
    if judge_model == "openai":
        messages = [
            {
                "role": "system",
                "content": "You are the Judge. Follow the user's instructions and respond in the requested format.",
            },
            {
                "role": "user",
                "content": judge_instructions + "\n\n" + debate_context,
            },
        ]
        judge_raw = call_openai_debater(messages, temperature=0.3, max_tokens=220)
        judge_title = "Judge – OpenAI (gpt-4.1-mini)"
    else:
        prompt = judge_instructions + "\n\n" + debate_context
        judge_raw = call_gemini_judge(prompt, temperature=0.3, max_tokens=220)
        judge_title = "Judge – Gemini (gemini-2.0-flash-lite)"

    # ---- Parse Winner / Reason / Final ----

    text = judge_raw.replace("\r", "")
    lower = text.lower()

    def _extract(label: str) -> str:
        """
        Extract content after '<label>:' until the next label or end.
        """
        idx = lower.find(label.lower() + ":")
        if idx == -1:
            return ""
        start = idx + len(label) + 1  # skip 'label:'
        rest = text[start:]
        # cut at next label if present
        for other in ["winner:", "reason:", "final:"]:
            if other.lower().startswith(label.lower()):
                continue
            pos = rest.lower().find(other)
            if pos != -1:
                rest = rest[:pos]
        return rest.strip(" \n-:")

    winner_raw = _extract("winner")
    reason_raw = _extract("reason")
    final_raw = _extract("final")

    if not final_raw:
        final_raw = text.strip()

    # Map A/B/tie to model names
    label_a = _model_human_name(model_a)
    label_b = _model_human_name(model_b)
    winner_display = "Unknown"

    wl = (winner_raw or "").lower()
    if "tie" in wl:
        winner_display = "Tie"
    elif "b" in wl and "a" not in wl:
        winner_display = label_b
    elif "a" in wl and "b" not in wl:
        winner_display = label_a

    state["winner"] = winner_display
    state["final_answer"] = final_raw
    state["judge_raw"] = judge_raw

    # Build nicely colored lines inside judge block
    winner_line = (
        f'<p style="margin:4px 0;">'
        f'<span style="color:#1b5e20; font-weight:600;">Winner:</span> '
        f'{winner_display or winner_raw or "Unknown"}</p>'
    )

    reason_line = ""
    if reason_raw:
        reason_line = (
            f'<p style="margin:4px 0;">'
            f'<span style="color:#0d47a1; font-weight:600;">Reason:</span> '
            f'{reason_raw}</p>'
        )

    final_line = (
        f'<p style="margin:4px 0;">'
        f'<span style="color:#e65100; font-weight:600;">Final:</span> '
        f'{final_raw}</p>'
    )

    judge_body = winner_line + reason_line + final_line
    judge_block = _html_block(JUDGE_COLOR, judge_title, judge_body)

    sections = state["transcript_sections"]
    sections.append("<h2>⚖️ Judge's Summary</h2>")
    sections.append(judge_block)
    state["transcript_sections"] = sections

    return state


def node_assemble(state: DebateState) -> DebateState:
    """
    Combine transcript_sections into a single HTML string.
    """

    transcript = "\n\n".join(state.get("transcript_sections", []))
    state["transcript_markdown"] = transcript
    return state
