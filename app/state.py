"""
state.py

Defines the state structure that flows through the LangGraph.
"""

from typing import TypedDict, List


class DebateState(TypedDict, total=False):
    # Input
    question: str
    temperature: float

    # Retrieved memory snippets (short past debates)
    memory_snippets: List[str]

    # Outputs from debaters
    opening_a: str
    opening_b: str
    rebuttals_a: List[str]
    rebuttals_b: List[str]

    # Judge results
    winner: str
    final_answer: str
    judge_raw: str

    # For UI display
    transcript_sections: List[str]
    transcript_markdown: str
