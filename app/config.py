import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()

class Settings(BaseModel):
    database_url: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    embedding_dim: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "384")))

settings = Settings()
