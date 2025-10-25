from pydantic import BaseModel, Field
from typing import Optional, List

# ----- Requests -----
class SimilarRequest(BaseModel):
    ticket_id: int
    top_k: int = Field(default=5, ge=1, le=50)

class AssignRequest(BaseModel):
    ticket_id: int
    top_k: int = Field(default=5, ge=1, le=50)

# ----- Responses -----
class SimilarItem(BaseModel):
    ticket_id: int
    chunk_id: int
    score: float
    title: Optional[str] = None
    answer: Optional[str] = None
    assigned_team_id: Optional[str] = None
    assigned_team_name: Optional[str] = None

class SimilarResponse(BaseModel):
    results: List[SimilarItem]

class AssignResponse(BaseModel):
    ticket_id: int
    assigned_team_id: str
    assigned_team_name: str
    reasoning: str
    persisted: bool
    message: Optional[str] = None



class SolutionRequest(BaseModel):
    ticket_id: int
    top_k: int = Field(default=5, ge=1, le=50)  

class SolutionSource(BaseModel):
    ticket_id: int
    title: Optional[str] = None
    score: float

class SolutionResponse(BaseModel):
    ticket_id: int
    solution: str
    sources: List[SolutionSource]
    persisted: bool                     
    message: Optional[str] = None       