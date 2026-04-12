"""
VIBE – FastAPI Server
OpenEnv-compliant endpoints: /reset, /step, /state
"""

import uvicorn
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


# ── Score safety (defensive copy here too) ────────────────────────────────────

def safe_score(score: float) -> float:
    """Strictly between (0, 1) — 0.0 and 1.0 are NOT valid."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0.05
    if s <= 0.0:
        return 0.01
    if s >= 1.0:
        return 0.99
    return round(s, 4)


# ── Request/Response schemas ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: str = "easy"


class StepRequest(BaseModel):
    decision: str
    reason:   str = ""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
@app.get("/")
def health():
    return {"status": "ok", "name": "VIBE", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global _env
    _env = AISafetyEnv(difficulty=req.difficulty)
    obs = _env.reset()
    data = obs.model_dump()
    # Guarantee reward is never 0.0 or 1.0 even if Observation default drifts
    data["reward"] = safe_score(data.get("reward", 0.05))
    return data


@app.post("/step")
def step(req: StepRequest):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    action = Action(decision=req.decision, reason=req.reason)
    result = _env.step(action)

    # ── CRITICAL: clamp every score/reward field before it leaves the server ──
    clamped_score  = safe_score(result.score)
    clamped_reward = safe_score(result.reward)

    obs_data = result.observation.model_dump()
    obs_data["reward"] = safe_score(obs_data.get("reward", 0.05))

    return {
        "score":       clamped_score,
        "reward":      clamped_reward,
        "done":        result.done,
        "info":        result.info,
        "observation": obs_data,
    }


@app.get("/state")
def state():
    global _env
    if _env is None:
        return {"status": "not_started"}
    return _env.state()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
