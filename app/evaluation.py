# app/evaluation.py

import os
from typing import Dict, List
from sqlalchemy import select
from openai import OpenAI

from .models import Ticket
from .repository import TicketDataAgent
from .assignment_agent import AssignmentAgent
from .solution_agent import SolutionAgent

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))



def run_assignment_evaluation(
    repository: TicketDataAgent,
    assign_agent: AssignmentAgent,
    limit: int = 3000,
    top_k: int = 5,
) -> Dict:
    """
    Evaluate the AssignmentAgent against existing tickets that already have
    an actual assigned_team_id.

    For each ticket:
      - Ensure it is indexed (embeddings exist)
      - Run the assignment agent (LLM)
      - Compare predicted team_id vs. actual assigned_team_id

    Returns a dict with:
      - total_evaluated
      - correct
      - incorrect
      - accuracy (percentage)
    """

    session = repository.session

    # 1) Get up to `limit` tickets that already have assigned_team_id
    rows = (
        session.execute(
            select(Ticket).where(Ticket.assigned_team_id.is_not(None)).limit(limit)
        )
        .scalars()
        .all()
    )

    total = len(rows)
    correct = 0
    incorrect = 0

    if total == 0:
        return {
            "total_evaluated": 0,
            "correct": 0,
            "incorrect": 0,
            "accuracy": 0.0,
        }

    for t in rows:
        ticket_id = t.ticket_id
        subject = t.subject or ""
        body = t.body or ""
        text = (subject + " " + body).strip()

        # Skip tickets with no text at all
        if not subject and not body:
            total -= 1
            continue

        # skip super-short / useless tickets
        if len(text) < 20:
            total -= 1
            continue

        # 2) Ensure embeddings exist for this ticket
        try:
            repository.ensure_indexed(ticket_id)
        except Exception as e:
            print(f"[EVAL] Failed to index ticket {ticket_id}: {e}")

        # 3) Run assignment agent (LLM prediction)
        try:
            result = assign_agent.assign_team(
                ticket_id=ticket_id,
                subject=subject,
                body=body,
                top_k=top_k,
            )
        except Exception as e:
            print(f"[EVAL] Assignment agent failed for ticket {ticket_id}: {e}")
            incorrect += 1
            continue

        predicted_team_id = (result or {}).get("assigned_team_id")

        if not predicted_team_id:
            incorrect += 1
        elif predicted_team_id == t.assigned_team_id:
            correct += 1
        else:
            incorrect += 1

    if total <= 0:
        accuracy = 0.0
    else:
        accuracy = (correct / total) * 100.0

    return {
        "total_evaluated": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy,
    }



def run_solution_evaluation(
    repository: TicketDataAgent,
    solution_agent: SolutionAgent,
    limit: int = 500,
    top_k: int = 20,
    judge_model: str = "gpt-4o-mini",
) -> Dict:
    """
    Evaluate the SolutionAgent against existing tickets that already have
    an actual 'answer' (reference solution) in the database.

    For each ticket:
      - Ensure it is indexed (embeddings exist) if needed
      - Generate a suggested solution with SolutionAgent
      - Use an LLM judge to compare suggested vs reference solution

    Returns a summary dict with:
      - total_evaluated
      - avg_similarity (0..1)
      - avg_similarity_percent
      - good_match
      - partial_match
      - mismatch
      - failed (LLM or generation errors)
    """

    session = repository.session

    # 1) Get up to `limit` tickets that already have a reference solution
    rows = (
        session.execute(
            select(Ticket).where(Ticket.answer.is_not(None)).limit(limit)
        )
        .scalars()
        .all()
    )

    total = len(rows)
    if total == 0:
        return {
            "total_evaluated": 0,
            "avg_similarity": 0.0,
            "avg_similarity_percent": 0.0,
            "good_match": 0,
            "partial_match": 0,
            "mismatch": 0,
            "failed": 0,
        }

    sum_similarity = 0.0
    good = 0
    partial = 0
    bad = 0
    failed = 0

    for idx, t in enumerate(rows, start=1):
        ticket_id = t.ticket_id
        subject = t.subject or ""
        body = t.body or ""
        reference_solution = (t.answer or "").strip()

        # Skip tickets with no useful text or no solution
        text_len = len((subject + " " + body).strip())
        if text_len < 20 or not reference_solution:
            total -= 1
            continue

        print(f"[SOLUTION EVAL] Processing ticket #{idx} (ticket_id={ticket_id})")

        # 2) Ensure indexed (optional but aligned with how you use SolutionAgent)
        try:
            repository.ensure_indexed(ticket_id)
        except Exception as e:
            print(f"[SOLUTION EVAL] Failed to index ticket {ticket_id}: {e}")

        # 3) Generate solution via SolutionAgent
        try:
            sol_result = solution_agent.generate_solution(
                ticket_id=ticket_id,
                subject=subject,
                body=body,
                top_k=top_k,
            )
        except Exception as e:
            print(f"[SOLUTION EVAL] SolutionAgent failed for ticket {ticket_id}: {e}")
            failed += 1
            continue

        generated_solution = sol_result.get("solution") or sol_result.get("answer") or ""
        generated_solution = generated_solution.strip()

        if not generated_solution:
            print(f"[SOLUTION EVAL] Empty generated solution for ticket {ticket_id}")
            failed += 1
            continue

        # 4) Ask LLM judge to compare
        try:
            judge = llm_grade_solution(
                ticket_id=ticket_id,
                subject=subject,
                body=body,
                reference_solution=reference_solution,
                generated_solution=generated_solution,
                model=judge_model,
            )
        except Exception as e:
            print(f"[SOLUTION EVAL] Judge LLM failed for ticket {ticket_id}: {e}")
            failed += 1
            continue

        similarity = judge.get("similarity", 0.0)
        category = judge.get("category", "mismatch")

        sum_similarity += similarity

        if category == "good_match":
            good += 1
        elif category == "partial_match":
            partial += 1
        else:
            bad += 1

    if total <= 0:
        avg_similarity = 0.0
    else:
        avg_similarity = sum_similarity / total

    return {
        "total_evaluated": total,
        "avg_similarity": avg_similarity,
        "avg_similarity_percent": avg_similarity * 100.0,
        "good_match": good,
        "partial_match": partial,
        "mismatch": bad,
        "failed": failed,
    }


def llm_grade_solution(
    ticket_id: int,
    subject: str,
    body: str,
    reference_solution: str,
    generated_solution: str,
    model: str = "gpt-4o-mini",
) -> Dict:
    """
    Ask the LLM to judge how close the generated solution is to the reference solution.

    Returns a dict like:
      {
        "similarity": 0.85,        # 0–1
        "category": "good_match",  # good_match | partial_match | mismatch
        "explanation": "..."
      }
    """

    # Basic safety defaults
    reference_solution = (reference_solution or "").strip()
    generated_solution = (generated_solution or "").strip()
    subject = subject or ""
    body = body or ""

    # If anything is empty, just short-circuit
    if not reference_solution or not generated_solution:
        return {
            "similarity": 0.0,
            "category": "mismatch",
            "explanation": "One of the solutions is empty; cannot compare meaningfully.",
        }

    prompt = """
        You are an IT service evaluation expert tasked with determining how accurately a suggested solution matches the actual resolution steps.

        Your job is to compare two solutions for the SAME ticket:
        1) The existing reference solution (what the support team actually used).
        2) The newly suggested solution (generated by an AI model).

        Your task:
        - Compare an existing (reference) solution with a newly suggested solution for the same ticket.
        - Focus on whether the suggested solution would correctly resolve the user's issue.
        - Ignore wording/phrasing differences; consider meaning and key steps.
        Compare the two lists:
        - Are the same main actions present?



        You must output a strict JSON object with these fields ONLY:
        - "similarity": a number between 0 and 1 indicating how close the suggested solution is to the reference in meaning and effectiveness.
            - 1.0 = essentially the same solution
            - 0.8 = very similar / acceptable alternative
            - 0.5 = partially correct, but missing important parts
            - 0.2 or below = mostly incorrect or unrelated
        - "category": one of "good_match", "partial_match", "mismatch"
            - good_match  -> similarity >= 0.6
            - partial_match -> 0.3 <= similarity < 0.6
            - mismatch -> similarity < 0.3

        Be strict and consistent:
        - Do NOT give a high similarity score just because both solutions mention the same system or application.
        """.strip()


    user_msg = f"""
        Ticket ID: {ticket_id}

        Ticket subject:
        {subject}

        Ticket body:
        {body}

        Reference solution (what was actually used to resolve the ticket):
        {reference_solution}

        Suggested solution (generated by the model):
        {generated_solution}

        Now evaluate how close the suggested solution is to the reference solution.
        """.strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,  # deterministic for evaluation
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content

    # Very simple parse; we expect proper JSON due to response_format
    import json
    try:
        data = json.loads(content)
    except Exception:
        # Fallback if something weird happens
        return {
            "similarity": 0.0,
            "category": "mismatch",
            "explanation": f"Could not parse judge response: {content}",
        }

    # Normalize & guard fields
    similarity = float(data.get("similarity", 0.0))
    similarity = max(0.0, min(1.0, similarity))

    category = (data.get("category") or "").strip().lower()
    if category not in ("good_match", "partial_match", "mismatch"):
        # simple categorization if needed
        if similarity >= 0.8:
            category = "good_match"
        elif similarity >= 0.4:
            category = "partial_match"
        else:
            category = "mismatch"

    explanation = data.get("explanation") or ""

    return {
        "similarity": similarity,
        "category": category,
        "explanation": explanation,
    }
