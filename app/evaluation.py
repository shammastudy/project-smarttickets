# app/evaluation.py

from typing import Dict
from sqlalchemy import select

from .models import Ticket
from .data_agent import TicketDataAgent
from .assignment_agent import AssignmentAgent


def run_assignment_evaluation(
    data_agent: TicketDataAgent,
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

    session = data_agent.session

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
            data_agent.ensure_indexed(ticket_id)
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
