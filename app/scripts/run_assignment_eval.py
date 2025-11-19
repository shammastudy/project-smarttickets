# scripts/run_assignment_eval.py
from app.repository import TicketDataAgent
from app.assignment_agent import AssignmentAgent
from app.db import engine
from app.evaluation import run_assignment_evaluation

if __name__ == "__main__":
    repository = TicketDataAgent()
    assign_agent = AssignmentAgent(engine)

    stats = run_assignment_evaluation(
        repository=repository,
        assign_agent=assign_agent,
        limit=1000,   # start small!
        top_k=10,
    )

    print("Evaluation stats:")
    for k, v in stats.items():
        print(f"{k}: {v}")
