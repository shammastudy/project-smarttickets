from fastapi import FastAPI, HTTPException
import app.schemas as schemas

from app.data_agent import TicketDataAgent
from app.assignment_agent import AssignmentAgent
from app.solution_agent import SolutionAgent
from app.retriever import embed_text, top_k_similar
from app.db import engine

app = FastAPI(title="Smart Tickets – API", version="1.0.0")

data_agent = TicketDataAgent()
assign_agent = AssignmentAgent(engine)
solution_agent = SolutionAgent()

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- SIMILAR (by ticket_id) ----------
@app.post("/similar", response_model=schemas.SimilarResponse)
def similar(req: schemas.SimilarRequest) -> schemas.SimilarResponse:
    # 1) fetch ticket text
    t = data_agent.get_ticket_text(req.ticket_id)
    if not t:
        raise HTTPException(status_code=404, detail=f"ticket_id {req.ticket_id} not found")

    subject = t.get("subject") or ""
    body = t.get("body") or ""
    text = f"Subject: {subject}\nBody: {body}".strip()
    if not text:
        return SimilarResponse(results=[])

    # 2) ensure embeddings exist for this ticket (safe no-op if already indexed)
    try:
        data_agent.ensure_indexed(req.ticket_id)
    except Exception as e:
        # not fatal; proceed anyway
        print(f"Indexing skipped/failed for ticket {req.ticket_id}: {e}")

    # 3) embed & search
    qvec = embed_text(text)
    rows = top_k_similar(qvec, top_k=req.top_k, exclude_ticket_id=req.ticket_id)  # ✅ pass exclude id
    return SimilarResponse(results=[SimilarItem(**r) for r in rows])


# ---------- ASSIGN (by ticket_id) ----------
@app.post("/assign", response_model=schemas.AssignResponse)
def assign(req: schemas.AssignRequest) -> schemas.AssignResponse:
    # 1) fetch ticket text
    t = data_agent.get_ticket_text(req.ticket_id)
    if not t:
        raise HTTPException(status_code=404, detail=f"ticket_id {req.ticket_id} not found")

    subject = t.get("subject") or ""
    body = t.get("body") or ""

    # 2) ensure embeddings exist for this ticket (optional safeguard)
    try:
        data_agent.ensure_indexed(req.ticket_id)
    except Exception as e:
        print(f"Indexing skipped/failed for ticket {req.ticket_id}: {e}")

    # 3) run assignment (LLM + strict team validation + retry)
    result = assign_agent.assign_team(req.ticket_id, subject, body, req.top_k)
    assigned_team_id = result.get("assigned_team_id") or ""
    assigned_team_name = result.get("assigned_team_name") or ""
    reasoning = result.get("reasoning") or "No reasoning provided."

    # 4) persist suggestion if valid
    if not assigned_team_id:
        return AssignResponse(
            ticket_id=req.ticket_id,
            assigned_team_id="",
            assigned_team_name="Unassigned",
            reasoning=reasoning,
            persisted=False,
            message="No valid team_id returned; not persisted."
        )

    persisted = data_agent.update_suggested_team(req.ticket_id, assigned_team_id)

    return AssignResponse(
        ticket_id=req.ticket_id,
        assigned_team_id=assigned_team_id,
        assigned_team_name=assigned_team_name,
        reasoning=reasoning,
        persisted=bool(persisted),
        message="suggested_assigned_team_id updated." if persisted else "Could not persist; check ticket_id/team_id."
    )


@app.post("/solution", response_model=schemas.SolutionResponse)
def solution(req: schemas.SolutionRequest) -> schemas.SolutionResponse:
    # 1) fetch subject/body
    t = data_agent.get_ticket_text(req.ticket_id)
    if not t:
        raise HTTPException(status_code=404, detail=f"ticket_id {req.ticket_id} not found")
    subject = t.get("subject") or ""
    body = t.get("body") or ""

    # 2) (optional) ensure indexed
    try:
        data_agent.ensure_indexed(req.ticket_id)
    except Exception as e:
        print(f"Indexing skipped/failed for ticket {req.ticket_id}: {e}")

    # 3) generate solution with RAG
    result = solution_agent.generate_solution(
        ticket_id=req.ticket_id,
        subject=subject,
        body=body,
        top_k=req.top_k
    )
    solution_text = result.get("solution", "No solution generated.")
    sources = [SolutionSource(**s) for s in result.get("sources", [])]

    # 4) persist suggested_answer
    persisted = data_agent.update_suggested_answer(req.ticket_id, solution_text)

    return SolutionResponse(
        ticket_id=req.ticket_id,
        solution=solution_text,
        sources=sources,
        persisted=bool(persisted),
        message="suggested_answer updated." if persisted else "Could not persist; ticket not found."
    )



# ---------- CREATE TICKET ----------
@app.post("/tickets", response_model=schemas.CreateTicketResponse)
def create_ticket(req: schemas.CreateTicketRequest) -> schemas.CreateTicketResponse:
    try:
        payload = req.model_dump(exclude_unset=True)

        # 🚫 Remove team fields if not provided or blank (prevents FK violations)
        for fk in ("assigned_team_id", "suggested_assigned_team_id"):
            if fk in payload and (payload[fk] is None or str(payload[fk]).strip() == ""):
                payload.pop(fk, None)

        ticket = data_agent.create_ticket(**payload)
        indexed_chunks = data_agent.count_ticket_embeddings(ticket.ticket_id)

        return CreateTicketResponse(
            ticket_id=ticket.ticket_id,
            indexed_chunks=indexed_chunks,
            message="Ticket created and indexed." if indexed_chunks > 0 else "Ticket created (no body or no chunks)."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create ticket: {e}")




# ------- Notifications Part --------

from pydantic import BaseModel, EmailStr
from typing import Optional
from app.mailer import (
    Mailer,
    tpl_ticket_submitted_user,
    tpl_new_ticket_received_hd,
    tpl_ticket_assigned_user,
    tpl_ticket_assigned_team,
    tpl_ticket_resolved_user,
    tpl_ticket_canceled_user,
)
from app.config import settings
mailer = Mailer()



# ---------- NOTIFICATIONS ----------
# 1) Ticket Submitted Successfully – to user (status: OPEN)
@app.post("/notify/ticket-submitted")
def notify_ticket_submitted(payload: schemas.TicketSubmittedUserPayload):
    try:
        subj, html = tpl_ticket_submitted_user(payload.ticket_id, payload.user_name)
        result = mailer.send_email(to=payload.recipient, subject=subj, html_body=html, cc=settings.DEPARTMENT_EMAIL)
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Mail send failed: {e}")


# 2) New Ticket Received – to helpdesk (status: OPEN)
@app.post("/notify/new-ticket-received")
def notify_new_ticket_received(payload: schemas.NewTicketReceivedPayload):
    to = payload.helpdesk_email or settings.DEPARTMENT_EMAIL
    if not to:
        raise HTTPException(status_code=400, detail="No helpdesk_email provided and DEPARTMENT_EMAIL is not set.")
    try:
        subj, html = tpl_new_ticket_received_hd(payload.ticket_id)
        result = mailer.send_email(to=to, subject=subj, html_body=html)
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Mail send failed: {e}")


# 3) Ticket Assigned – To user (status: ASSIGNED)
@app.post("/notify/ticket-assigned-user")
def notify_ticket_assigned_user(payload: schemas.TicketAssignedUserPayload):
    try:
        subj, html = tpl_ticket_assigned_user(payload.ticket_id, payload.user_name, payload.team_name)
        result = mailer.send_email(to=payload.recipient, subject=subj, html_body=html, cc=settings.DEPARTMENT_EMAIL)
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Mail send failed: {e}")


# 4) New Ticket Assigned to Your Team – To support team (status: ASSIGNED)
@app.post("/notify/ticket-assigned-team")
def notify_ticket_assigned_team(payload: schemas.TicketAssignedTeamPayload):
    to = payload.team_email or settings.DEPARTMENT_EMAIL
    if not to:
        raise HTTPException(status_code=400, detail="No team_email provided and DEPARTMENT_EMAIL is not set.")
    try:
        subj, html = tpl_ticket_assigned_team(payload.ticket_id, payload.team_name)
        result = mailer.send_email(to=to, subject=subj, html_body=html)
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Mail send failed: {e}")


# 5) Ticket Resolved – To user (status: RESOLVED)
@app.post("/notify/ticket-resolved")
def notify_ticket_resolved(payload: schemas.TicketResolvedUserPayload):
    try:
        subj, html = tpl_ticket_resolved_user(payload.ticket_id, payload.user_name)
        result = mailer.send_email(to=payload.recipient, subject=subj, html_body=html, cc=settings.DEPARTMENT_EMAIL)
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Mail send failed: {e}")


# 6) Ticket Canceled – To user (status: CANCELED)
@app.post("/notify/ticket-canceled")
def notify_ticket_canceled(payload: schemas.TicketCanceledUserPayload):
    try:
        subj, html = tpl_ticket_canceled_user(payload.ticket_id, payload.user_name)
        result = mailer.send_email(to=payload.recipient, subject=subj, html_body=html, cc=settings.DEPARTMENT_EMAIL)
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Mail send failed: {e}")

