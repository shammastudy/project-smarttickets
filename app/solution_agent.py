import os, re, json
from typing import List, Dict
from openai import OpenAI
from app.retriever import top_k_similar, embed_text

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

import re

CONTACT_RE = re.compile(r"(contact|call|email|reach\s*out|open a ticket|service desk)", re.I)

def _is_actionable(answer: str | None) -> bool:
    if not answer:
        return False
    # at least some substance and not just “contact support”
    if len(answer.strip()) < 40:
        return False
    if CONTACT_RE.search(answer):
        return False
    return True

def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|md|markdown)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _safe_parse_json(s: str) -> dict:
    s = _strip_fences(s)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"solution": s}

class SolutionAgent:
    """
    Generates a solution for a ticket using RAG:
    - embeds the new ticket (subject+body)
    - retrieves similar tickets (excluding the same ticket id)
    - prompts the LLM to synthesize a solution
    """
    def __init__(self):
        pass

    def generate_solution(self, ticket_id: int, subject: str, body: str, top_k: int = 5) -> dict:
        query_text = f"Subject: {subject or ''}\nBody: {body or ''}".strip()
        if not query_text:
            return {
                "solution": "No subject/body found for this ticket. Please provide details.",
                "sources": []
            }

        # Retrieve neighbors (exclude the same ticket_id)
        qvec = embed_text(query_text)
        sims = top_k_similar(qvec, top_k=top_k, exclude_ticket_id=ticket_id)

        # Build concise context from neighbors (keep only actionable answers)
        lines, sources = [], []
        for r in sims:
            ans = r.get("answer") or ""
            if not _is_actionable(ans):
                continue
            lines.append(
                f"TicketID: {r.get('ticket_id')}\n"
                f"Title: {r.get('title') or 'N/A'}\n"
                f"Answer: {ans}\n"
                f"---"
            )
            sources.append({
                "ticket_id": r.get("ticket_id"),
                "title": r.get("title"),
                "score": r.get("score"),
            })

        context = "\n".join(lines) if lines else "No actionable prior solutions."

        prompt = f"""
            You are a helpdesk *solution* assistant. Your role is to generate a clear, concise, and directly actionable fix plan for the NEW ticket based on relevant past solutions.

            NEW TICKET DETAILS:
            {query_text}

            RELEVANT PRIOR SOLUTIONS (already filtered for actionable content):
            {context}

            GUIDELINES (follow exactly):
            - Write a short, practical solution — 3–5 numbered steps maximum.
            - Use short, direct sentences (avoid long or complex phrasing).
            - Focus only on steps that can be executed immediately by a user or L1/L2 technician.
            - Prefer fixes confirmed to work in prior solutions.
            - For commands or settings, include exact menu paths or commands.
            - Add brief verification after key steps (how to confirm the issue is resolved).
            - Avoid unnecessary details or long explanations.
            - Do **not** include “contact support”, phone numbers, or generic escalation text.
            - If escalation is required, add a final **Escalate if:** section listing specific triggers and which team should handle it.

            Respond **only** as a compact JSON object using this schema:
            {{"solution": "<markdown with numbered steps and short notes>"}}
            """


        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}  # remove if your account doesn't support it
        )
        raw = resp.choices[0].message.content
        parsed = _safe_parse_json(raw)

        # Guarantee schema
        solution = parsed.get("solution") or "No solution generated."
        return {
            "solution": solution,
            "sources": sources
        }
