"""
memory.py

ChromaDB-based memory for debates.

We:
- Store each debate as a short document.
- Retrieve similar past debates for a new question.
"""

import os
import uuid
from typing import List

import chromadb
from chromadb.utils import embedding_functions

from .config import OPENAI_API_KEY

# Path for local ChromaDB persistence (folder at repo root)
CHROMA_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "chroma_db",
)
COLLECTION_NAME = "debates"

# -------------------------------
# Initialize Chroma client + collection
# -------------------------------

# Use OpenAI embeddings (cheap, and you already have the key)
openai_embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",  # low-cost embedding model
)

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=openai_embedding_fn,
)


# -------------------------------
# Store memory
# -------------------------------

def store_debate_memory(
    question: str,
    final_answer: str,
    winner: str,
) -> None:
    """
    Create a short document for the debate and store it in Chroma.

    We keep it compact to save tokens and disk space.
    """
    question = (question or "").strip()
    final_answer = (final_answer or "").strip()

    if not question or not final_answer:
        return

    doc_id = str(uuid.uuid4())

    text = (
        f"Question: {question}\n"
        f"Winner: {winner}\n"
        f"Final answer:\n{final_answer}"
    )

    collection.add(
        ids=[doc_id],
        documents=[text],
        metadatas=[{"winner": winner}],
    )


# -------------------------------
# Load memory
# -------------------------------

def load_relevant_memories(
    question: str,
    top_k: int = 3,
) -> List[str]:
    """
    Retrieve up to top_k similar past debates for the given question.
    Returns a list of small text snippets.
    """

    question = (question or "").strip()
    if not question:
        return []

    try:
        results = collection.query(
            query_texts=[question],
            n_results=top_k,
        )
    except Exception:
        # If something goes wrong, we just return no memory.
        return []

    docs = results.get("documents", [[]])
    if not docs or not docs[0]:
        return []

    snippets: List[str] = []
    for doc in docs[0]:
        # Truncate each snippet to keep prompts small
        snippets.append(doc[:500])

    return snippets
