"""
VIBE – FastAPI Server
OpenEnv-compliant endpoints: /reset, /step, /state
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from environment.env import AISafetyEnv, Action, Observation, StepResult

app = FastAPI(
    title="VIBE – Veritas Integrity Behavior Engine",
    description="Context-aware content moderation OpenEnv environment",
    version="1.0.0",
)

# Global environment instance (one per server process)
_env: Optional[AISafetyEnv] = None


# ── Request/Response schemas ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: str = "easy"


class StepRequest(BaseModel):
    decision: str
    reason:   str = ""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "name": "VIBE", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global _env
    _env = AISafetyEnv(difficulty=req.difficulty)
    obs = _env.reset()
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    action = Action(decision=req.decision, reason=req.reason)
    result = _env.step(action)

    return {
        "score":         result.score,
        "reward":        result.reward,
        "done":          result.done,
        "info":          result.info,
        "observation":   result.observation.model_dump(),
    }


@app.get("/state")
def state():
    global _env
    if _env is None:
        return {"status": "not_started"}
    return _env.state()
def main():
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
 
 
if __name__ == "__main__":
    main()
