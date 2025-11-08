import os, re, json
from typing import List, Dict
from sqlalchemy import text
from openai import OpenAI
from app.retriever import top_k_similar, embed_text
from app.db import engine

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ------------------------------
# Helper functions
# ------------------------------
def safe_parse_json(text_in: str) -> dict:
    if not isinstance(text_in, str):
        return {"assigned_team_id": "", "assigned_team_name": "", "reasoning": "Non-string model output."}
    text_in = re.sub(r"^```(?:json)?\s*|\s*```$", "", text_in.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(text_in)
    except Exception:
        m = re.search(r"\{.*\}", text_in, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"assigned_team_id": "", "assigned_team_name": text_in.strip(), "reasoning": "Parsed from non-JSON output."}

def load_all_teams() -> List[Dict[str, str]]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT team_id, team_name FROM teams ORDER BY team_name")).mappings().all()
    return [{"team_id": r["team_id"], "team_name": r["team_name"]} for r in rows]

def normalize(s: str) -> str:
    return (s or "").strip().lower()


# ------------------------------
# Assignment Agent Class
# ------------------------------
class AssignmentAgent:
    def __init__(self, engine):
        self.engine = engine
        print("✅ AssignmentAgent ready (with retry mechanism)")

    # ✅ Main public method
    def assign_team(self, ticket_id:int, subject: str, body: str, top_k: int = 5):
        # 1️⃣ Load valid teams from DB
        candidates = load_all_teams()
        if not candidates:
            return {
                "assigned_team_id": "",
                "assigned_team_name": "Unassigned",
                "reasoning": "No teams found in database."
            }

        # 2️⃣ Retrieve similar tickets
        query_text = f"Subject: {subject}\nBody: {body}"
        qvec = embed_text(query_text)
        sims = top_k_similar(qvec, top_k=top_k, exclude_ticket_id=ticket_id)

        # 3️⃣ Build context for prompt
        examples = []
        for t in sims:
            team_label = t.get("assigned_team_name") or "Unknown"
            examples.append(
                f"Title: {t.get('title')}\n"
                f"Answer: {t.get('answer') or 'N/A'}\n"
                f"Team: {team_label}\n---"
            )
        examples_text = "\n".join(examples) if examples else "No prior examples."

        # 4️⃣ Build team list JSON
        options_json = json.dumps(candidates, ensure_ascii=False)

        # 5️⃣ Compose the initial LLM prompt
        prompt = f"""
You are a helpdesk employee, responsible for assigning tickets to different teams with different specialities. Choose the correct support team **only** from the provided list.

New ticket:
{query_text}

Similar resolved tickets (with their teams):
{examples_text}

Valid teams (choose exactly one by ID and name):
{options_json}

Return ONLY a strict JSON object with these EXACT keys:
{{"assigned_team_id": "<id from list>", "assigned_team_name": "<matching name from list>", "reasoning": "<short reason>"}}
No extra text, no code fences, no markdown.
""".strip()

        # 6️⃣ Call the LLM
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        raw = resp.choices[0].message.content
        parsed = safe_parse_json(raw)

        # 7️⃣ Validate result
        valid_result = self._validate_or_retry(parsed, candidates, query_text)
        return valid_result

    # ------------------------------
    # ✅ NEW helper method for retry logic
    # ------------------------------
    def _retry_llm_assignment(self, candidates, query_text):
        """Ask the LLM again with a stronger prompt if first choice was invalid."""
        print("⚠️ Model selected an invalid team — retrying once with explicit team list...")

        retry_prompt = f"""
The previous response contained an invalid team name.

You must select one valid team **only** from this list:
{json.dumps(candidates, ensure_ascii=False)}

New ticket:
{query_text}

Return ONLY a strict JSON object:
{{"assigned_team_id": "<id from list>", "assigned_team_name": "<matching name>", "reasoning": "<short reason>"}}
""".strip()

        retry_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": retry_prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        retry_raw = retry_resp.choices[0].message.content
        retry_parsed = safe_parse_json(retry_raw)
        return retry_parsed

    # ------------------------------
    # ✅ Validation and Retry Wrapper
    # ------------------------------
    def _validate_or_retry(self, parsed, candidates, query_text):
        """Validate the LLM response; if invalid, retry once."""
        want_id = normalize(parsed.get("assigned_team_id", ""))
        want_name = normalize(parsed.get("assigned_team_name", ""))

        # Check if team exists in DB
        by_id = next((t for t in candidates if normalize(t["team_id"]) == want_id), None)
        by_name = next((t for t in candidates if normalize(t["team_name"]) == want_name), None)

        if by_id:
            return {
                "assigned_team_id": by_id["team_id"],
                "assigned_team_name": by_id["team_name"],
                "reasoning": parsed.get("reasoning", "Selected by ID.")
            }

        if by_name:
            return {
                "assigned_team_id": by_name["team_id"],
                "assigned_team_name": by_name["team_name"],
                "reasoning": parsed.get("reasoning", "Selected by name.")
            }

        # 🚀 Retry once if invalid
        retry_parsed = self._retry_llm_assignment(candidates, query_text)

        want_id2 = normalize(retry_parsed.get("assigned_team_id", ""))
        want_name2 = normalize(retry_parsed.get("assigned_team_name", ""))

        by_id2 = next((t for t in candidates if normalize(t["team_id"]) == want_id2), None)
        by_name2 = next((t for t in candidates if normalize(t["team_name"]) == want_name2), None)

        if by_id2:
            return {
                "assigned_team_id": by_id2["team_id"],
                "assigned_team_name": by_id2["team_name"],
                "reasoning": retry_parsed.get("reasoning", "Valid team found after retry (by ID).")
            }

        if by_name2:
            return {
                "assigned_team_id": by_name2["team_id"],
                "assigned_team_name": by_name2["team_name"],
                "reasoning": retry_parsed.get("reasoning", "Valid team found after retry (by name).")
            }

        # 🚨 Still invalid → fallback
        print(f"❌ Retry failed. Model output: {retry_parsed}")
        return {
            "assigned_team_id": "",
            "assigned_team_name": "Unassigned",
            "reasoning": "Model failed twice to return a valid team from database."
        }
