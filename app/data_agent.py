from datetime import datetime
from sqlalchemy import select, text, func, delete, update as sa_update
from sqlalchemy.orm import Session
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .db import engine, SessionLocal
from .models import Base, Ticket, Team, TeamMember, User, TicketEmbedding

# Ensure tables exist (optional; comment if you run migrations)
Base.metadata.create_all(engine)

class IndexerAgent:
    def __init__(self, engine, embedder):
        self.engine = engine
        self.embedder = embedder
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    def index_ticket(self, ticket_id: int, text_value: str):
        chunks = self.splitter.split_text(text_value or "")
        if not chunks:
            return 0

        with self.engine.begin() as conn:
            for chunk in chunks:
                emb = self.embedder.embed_query(chunk)  # 384-d
                # pgvector psycopg2 can take Python list directly with Vector type
                conn.execute(
                    text("""
                        INSERT INTO ticket_embeddings (ticket_id, chunk_text, embedding)
                        VALUES (:ticket_id, :chunk_text, :embedding)
                    """),
                    {"ticket_id": ticket_id, "chunk_text": chunk, "embedding": emb}
                )
        return len(chunks)

class TicketDataAgent:
    def __init__(self, db_url: str | None = None):
        self.session: Session = SessionLocal()
        self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.indexer = IndexerAgent(engine, self.embedder)

    def create_ticket(self, **kwargs):
        def _nz(v):
            return None if v is None or (isinstance(v, str) and v.strip() == "") else v

        try:
            ticket = Ticket(
                ticket_id=kwargs.get("ticket_id"),
                requester_id=kwargs.get("requester_id"),
                subject=kwargs.get("subject"),
                body=kwargs.get("body"),
                answer=kwargs.get("answer"),
                suggested_answer=kwargs.get("suggested_answer"),
                type=kwargs.get("type"),
                priority=kwargs.get("priority"),
                assigned_team_id=_nz(kwargs.get("assigned_team_id")),            # will be None if missing/blank
                assigned_team_user_id=kwargs.get("assigned_team_user_id"),
                suggested_assigned_team_id=_nz(kwargs.get("suggested_assigned_team_id")),  # None if missing/blank
                status=kwargs.get("status", "open"),
                created_at=datetime.utcnow(),
            )
            tags = kwargs.get("tags") or []
            for i, tag in enumerate(tags[:8], start=1):
                setattr(ticket, f"tag_{i}", tag)

            self.session.add(ticket)
            self.session.commit()  # get ticket_id

            if ticket.body:
                self.indexer.index_ticket(ticket.ticket_id, ticket.body)
            return ticket

        except Exception:
            self.session.rollback()
            raise

    def read_ticket(self, ticket_id: int):
        t = self.session.query(Ticket).filter_by(ticket_id=ticket_id).first()
        if not t:
            return None
        return {
            "ticket_id": t.ticket_id,
            "requester_id": t.requester_id,
            "subject": t.subject,
            "body": t.body,
            "answer": t.answer,
            "suggested_answer": t.suggested_answer,
            "type": t.type,
            "priority": t.priority,
            "assigned_team_id": t.assigned_team_id,
            "assigned_team_user_id": t.assigned_team_user_id,
            "suggested_assigned_team_id": t.suggested_assigned_team_id,
            "status": t.status,
            "created_at": t.created_at,
            "tags": [getattr(t, f"tag_{i}") for i in range(1, 9)]
        }
    
    
    def update_suggested_team(self, ticket_id: int, team_id: str) -> bool:
        """
        Set tickets.suggested_assigned_team_id = team_id for the given ticket_id.
        Returns True if updated, False if ticket not found or team invalid.
        """
        # Validate team exists
        team = self.session.execute(
            select(Team).where(Team.team_id == team_id)
        ).scalar_one_or_none()
        if not team:
            return False

        # Find ticket
        ticket = self.session.execute(
            select(Ticket).where(Ticket.ticket_id == ticket_id)
        ).scalar_one_or_none()
        if not ticket:
            return False

        ticket.suggested_assigned_team_id = team_id
        self.session.commit()
        return True
    
    def get_ticket_text(self, ticket_id: int) -> dict | None:
        """
        Returns {"subject": str|None, "body": str|None} for a given ticket_id,
        or None if not found.
        """
        t = self.session.execute(
            select(Ticket.subject, Ticket.body).where(Ticket.ticket_id == ticket_id)
        ).first()
        if not t:
            return None
        return {"subject": t.subject, "body": t.body}

    def count_ticket_embeddings(self, ticket_id: int) -> int:
        """
        How many chunks/embeddings exist for this ticket?
        """
        res = self.session.execute(
            select(func.count(TicketEmbedding.id)).where(TicketEmbedding.ticket_id == ticket_id)
        ).scalar_one()
        return int(res or 0)

    def ensure_indexed(self, ticket_id: int) -> int:
        """
        If the ticket has a body and no embeddings yet, index it now.
        Returns number of chunks embedded (0 if already indexed or no body).
        """
        t = self.session.execute(
            select(Ticket.ticket_id, Ticket.body).where(Ticket.ticket_id == ticket_id)
        ).first()
        if not t:
            return 0
        if not t.body:
            return 0
        existing = self.count_ticket_embeddings(ticket_id)
        if existing > 0:
            return 0
        return self.indexer.index_ticket(ticket_id, t.body)
    
    def update_suggested_answer(self, ticket_id: int, solution_text: str) -> bool:
        """
        Store the generated solution into tickets.suggested_answer.
        Returns True if updated, False if ticket not found.
        """
        ticket = self.session.execute(
            select(Ticket).where(Ticket.ticket_id == ticket_id)
        ).scalar_one_or_none()
        if not ticket:
            return False

        ticket.suggested_answer = solution_text
        self.session.commit()
        return True
    
