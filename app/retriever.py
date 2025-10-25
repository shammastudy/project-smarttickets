from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from .db import engine
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import text


# Use the same sentence-transformers model (384-d)
_embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text_value: str) -> list[float]:
    return _embedder.embed_query(text_value or "")

def top_k_similar(qvec, top_k=5, exclude_ticket_id=None):
    qvec_str = "[" + ",".join(map(str, qvec)) + "]"

    query = text("""
        SELECT
            t.ticket_id,
            t.subject AS title,
            t.answer,
            t.assigned_team_id,
            tm.team_name AS assigned_team_name,
            te.id AS chunk_id,
            (te.embedding <-> (:qvec)::vector) AS score
        FROM ticket_embeddings te
        JOIN tickets t   ON te.ticket_id = t.ticket_id
        LEFT JOIN teams tm ON t.assigned_team_id = tm.team_id
        WHERE (:exclude_id IS NULL OR te.ticket_id <> :exclude_id)   -- ✅ explicit exclusion
        ORDER BY score ASC
        LIMIT :top_k;
    """)

    with engine.begin() as conn:
        rows = conn.execute(
            query,
            {"qvec": qvec_str, "top_k": top_k, "exclude_id": exclude_ticket_id}
        ).mappings().all()

    return [dict(r) for r in rows]

