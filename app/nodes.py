"""
nodes.py

Contains all LangGraph node functions:
- load_memory
- opening
- rebuttal_round_1
- rebuttal_round_2
- judge
- store_memory
- assemble

Each node takes a DebateState and returns an updated DebateState.
"""

from typing import List

from .state import DebateState
from .clients import (
    call_openai_debater,
    call_grok_debater,
    call_gemini_judge,
)
from .memory import load_relevant_memories, store_debate_memory


# -------------------------------
# Memory nodes
# -------------------------------

def node_load_memory(state: DebateState) -> DebateState:
    """
    Load relevant past debates from Chroma for the current question.
    """
    question = state["question"]
    snippets = load_relevant_memories(question, top_k=3)

    state["memory_snippets"] = snippets

    sections = state.get("transcript_sections", [])
    if snippets:
        sections.append("## 📝 Retrieved Memory from Past Debates")
        for i, snippet in enumerate(snippets, start=1):
            sections.append(f"**Memory {i}:**\n\n{snippet}")
    state["transcript_sections"] = sections

    return state


def node_store_memory(state: DebateState) -> DebateState:
    """
    Store the final result of this debate into Chroma for future use.
    """
    question = state.get("question", "")
    final_answer = state.get("final_answer", "")
    winner = state.get("winner", "Unknown")

    store_debate_memory(question, final_answer, winner)
    return state


# -------------------------------
# Helper functions for rebuttals
# -------------------------------

def _short_rebuttal_for_a(question: str, other_text: str, temperature: float) -> str:
    """
    Debater A (OpenAI) rebuts B in short bullet points.
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
    return call_openai_debater(messages, temperature=temperature, max_tokens=120)


def _short_rebuttal_for_b(question: str, other_text: str, temperature: float) -> str:
    """
    Debater B (Grok) rebuts A.
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
    return call_grok_debater(messages, temperature=temperature, max_tokens=120)


# -------------------------------
# Main node functions
# -------------------------------

def node_opening(state: DebateState) -> DebateState:
    """
    Both debaters give a short opening answer.
    If memory_snippets exist, they are provided as context.
    """

    question = state["question"]
    temperature = state["temperature"]
    memory_snippets: List[str] = state.get("memory_snippets", [])

    memory_context = ""
    if memory_snippets:
        memory_context = (
            "Here are some relevant past debates. You may reuse useful ideas, "
            "but do not copy sentences word-for-word:\n\n"
            + "\n\n---\n\n".join(memory_snippets)
        )
        # Truncate to avoid huge prompts
        memory_context = memory_context[:1200]

    # Debater A (OpenAI) – short opening
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
    opening_a = call_openai_debater(messages_a, temperature=temperature, max_tokens=220)

    # Debater B (Grok) – short opening
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
    opening_b = call_grok_debater(messages_b, temperature=temperature, max_tokens=220)

    # Initialize transcript
    sections = state.get("transcript_sections", [])
    sections.append(f"## 🧠 Question\n\n{question}")
    sections.append("## 🎙️ Opening Statements")
    sections.append("### Debater A (OpenAI)\n" + opening_a)
    sections.append("### Debater B (Grok)\n" + opening_b)

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

    opening_a = state["opening_a"]
    opening_b = state["opening_b"]

    rebuttal_a = _short_rebuttal_for_a(question, opening_b, temp)
    rebuttal_b = _short_rebuttal_for_b(question, opening_a, temp)

    state["rebuttals_a"].append(rebuttal_a)
    state["rebuttals_b"].append(rebuttal_b)

    sections = state["transcript_sections"]
    sections.append("## 🔁 Rebuttal Round 1")
    sections.append("### Debater A (OpenAI) – Rebuttal\n" + rebuttal_a)
    sections.append("### Debater B (Grok) – Rebuttal\n" + rebuttal_b)
    state["transcript_sections"] = sections

    return state


def node_rebuttal_round_2(state: DebateState) -> DebateState:
    """
    Second short rebuttal round.
    """

    question = state["question"]
    temp = state["temperature"]

    latest_a = state["rebuttals_a"][-1]
    latest_b = state["rebuttals_b"][-1]

    rebuttal_a2 = _short_rebuttal_for_a(question, latest_b, temp)
    rebuttal_b2 = _short_rebuttal_for_b(question, latest_a, temp)

    state["rebuttals_a"].append(rebuttal_a2)
    state["rebuttals_b"].append(rebuttal_b2)

    sections = state["transcript_sections"]
    sections.append("## 🔁 Rebuttal Round 2")
    sections.append("### Debater A (OpenAI) – Rebuttal\n" + rebuttal_a2)
    sections.append("### Debater B (Grok) – Rebuttal\n" + rebuttal_b2)
    state["transcript_sections"] = sections

    return state


def node_judge(state: DebateState) -> DebateState:
    """
    Gemini acts as the Judge.
    It sees:
    - Question
    - Opening from A and B
    - Latest rebuttal from A and B
    """

    question = state["question"]
    opening_a = state["opening_a"]
    opening_b = state["opening_b"]

    latest_a = state["rebuttals_a"][-1]
    latest_b = state["rebuttals_b"][-1]

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

    prompt = (
        f"{judge_instructions}\n\n"
        f"Question:\n{question}\n\n"
        f"Debater A (Opening):\n{opening_a}\n\n"
        f"Debater B (Opening):\n{opening_b}\n\n"
        f"Debater A (Latest Rebuttal):\n{latest_a}\n\n"
        f"Debater B (Latest Rebuttal):\n{latest_b}\n"
    )

    judge_raw = call_gemini_judge(prompt, temperature=0.3, max_tokens=220)

    # Very simple parsing of judge output
    winner = "Unknown"
    final_answer = ""

    lower = judge_raw.lower()
    if "winner:" in lower:
        part = lower.split("winner:", 1)[1][:20]
        if "a" in part:
            winner = "A (OpenAI)"
        if "b" in part:
            winner = "B (Grok)"
        if "tie" in part:
            winner = "Tie"

    if "final:" in judge_raw:
        final_answer = judge_raw.split("final:", 1)[1].strip()
    else:
        final_answer = judge_raw.strip()

    state["winner"] = winner
    state["final_answer"] = final_answer
    state["judge_raw"] = judge_raw

    sections = state["transcript_sections"]
    sections.append("## ⚖️ Judge's Summary (Gemini)")
    sections.append(judge_raw)
    state["transcript_sections"] = sections

    return state


def node_assemble(state: DebateState) -> DebateState:
    """
    Combine transcript_sections into a single Markdown string.
    """

    transcript = "\n\n".join(state.get("transcript_sections", []))
    state["transcript_markdown"] = transcript
    return state
