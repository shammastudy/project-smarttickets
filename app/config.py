# app/config.py
import os
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Loads .env into os.environ (ok to keep even with BaseSettings)
load_dotenv()

class Settings(BaseSettings):
    # --- Core settings ---
    database_url: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    embedding_model: str = Field(default_factory=lambda: os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    ))
    embedding_dim: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "384")))

    # --- SMTP / Mail settings (optional so the app can start without mail configured) ---
    MAIL_HOST: Optional[str] = None
    MAIL_PORT: int = 587
    MAIL_USERNAME: Optional[str] = None
    MAIL_PASSWORD: Optional[str] = None
    MAIL_FROM: Optional[str] = None
    MAIL_FROM_NAME: str = "Smart Tickets"
    MAIL_REPLY_TO: Optional[str] = None
    MAIL_TLS: bool = True      # STARTTLS (587)
    MAIL_SSL: bool = False     # SMTPS (465)

    # Optional: default department mailbox to cc on notifications
    DEPARTMENT_EMAIL: Optional[str] = None

    # Tell Pydantic to read from .env automatically
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
