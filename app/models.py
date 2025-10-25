from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime, ForeignKey
from datetime import datetime
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str | None] = mapped_column(String)
    email: Mapped[str | None] = mapped_column(String)
    tickets = relationship("Ticket", back_populates="requester")

class Team(Base):
    __tablename__ = "teams"
    team_id: Mapped[str] = mapped_column(String, primary_key=True)
    team_name: Mapped[str | None] = mapped_column(String)

class TeamMember(Base):
    __tablename__ = "team_members"
    team_member_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[str] = mapped_column(String, ForeignKey("teams.team_id"))
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.user_id"))

class Ticket(Base):
    __tablename__ = "tickets"
    ticket_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    requester_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.user_id"))
    subject: Mapped[str | None] = mapped_column(String)
    body: Mapped[str | None] = mapped_column(String)
    answer: Mapped[str | None] = mapped_column(String)
    suggested_answer: Mapped[str | None] = mapped_column(String)
    type: Mapped[str | None] = mapped_column(String)
    priority: Mapped[str | None] = mapped_column(String)
    assigned_team_id: Mapped[str | None] = mapped_column(String, ForeignKey("teams.team_id"))
    assigned_team_user_id: Mapped[int | None] = mapped_column(Integer)
    suggested_assigned_team_id: Mapped[str | None] = mapped_column(String)
    status: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    tag_1: Mapped[str | None] = mapped_column(String)
    tag_2: Mapped[str | None] = mapped_column(String)
    tag_3: Mapped[str | None] = mapped_column(String)
    tag_4: Mapped[str | None] = mapped_column(String)
    tag_5: Mapped[str | None] = mapped_column(String)
    tag_6: Mapped[str | None] = mapped_column(String)
    tag_7: Mapped[str | None] = mapped_column(String)
    tag_8: Mapped[str | None] = mapped_column(String)

    requester = relationship("User", back_populates="tickets")

class TicketEmbedding(Base):
    __tablename__ = "ticket_embeddings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[int] = mapped_column(Integer, ForeignKey("tickets.ticket_id"))
    chunk_text: Mapped[str | None] = mapped_column(String)
    embedding = mapped_column(Vector(384))
