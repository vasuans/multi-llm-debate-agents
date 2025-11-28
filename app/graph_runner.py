"""
graph_runner.py

Builds the LangGraph:
load_memory -> opening -> rebuttal1 -> rebuttal2 -> judge -> store_memory -> assemble -> END

Also exposes a run_debate() function that the UI (and tests) can call.
"""

from langgraph.graph import StateGraph, END

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


# Build the graph
graph = StateGraph(DebateState)

graph.add_node("load_memory", node_load_memory)
graph.add_node("opening", node_opening)
graph.add_node("rebuttal1", node_rebuttal_round_1)
graph.add_node("rebuttal2", node_rebuttal_round_2)
graph.add_node("judge", node_judge)
graph.add_node("store_memory", node_store_memory)
graph.add_node("assemble", node_assemble)

graph.set_entry_point("load_memory")
graph.add_edge("load_memory", "opening")
graph.add_edge("opening", "rebuttal1")
graph.add_edge("rebuttal1", "rebuttal2")
graph.add_edge("rebuttal2", "judge")
graph.add_edge("judge", "store_memory")
graph.add_edge("store_memory", "assemble")
graph.add_edge("assemble", END)

# Compile into a runnable object
compiled_graph = graph.compile()


def run_debate(question: str, creativity: float) -> tuple[str, str]:
    """
    Helper used by the UI.

    Returns:
        final_answer_markdown, full_transcript_markdown
    """

    question = question.strip()
    if not question:
        return "Please enter a question or topic.", ""

    initial_state: DebateState = {
        "question": question,
        "temperature": creativity,
        "transcript_sections": [],
        "memory_snippets": [],
    }

    result = compiled_graph.invoke(initial_state)

    winner = result.get("winner", "Unknown")
    final_answer = result.get("final_answer", "").strip()
    transcript_md = result.get("transcript_markdown", "")

    pretty_final = f"### 🏁 Final Answer (Winner: {winner})\n\n{final_answer}"
    return pretty_final, transcript_md
