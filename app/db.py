from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from .config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

# Ensure pgvector and column exist (safe to run each start)
def init_pgvector():
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
init_pgvector()
