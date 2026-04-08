"""
VIBE – AISafetyEnv
Core OpenEnv-compatible environment for context-aware content moderation.
"""

import random
from pydantic import BaseModel

from environment.dataset import SCENARIOS
from environment.graders import grade_easy, grade_medium, grade_hard


# ── Pydantic Models ──────────────────────────────────────────────────────────

class Observation(BaseModel):
    video_context: str
    chat_history: list
    comment: str
    difficulty: str
    reward: float = 0.0
    done: bool = False
    message: str = ""


class Action(BaseModel):
    decision: str
    reason: str = ""


class StepResult(BaseModel):
    score: float
    done: bool
    info: dict
    reward: float
    observation: Observation


# ── Environment ──────────────────────────────────────────────────────────────

class AISafetyEnv:

    def __init__(self, difficulty: str = "easy"):
        self.difficulty = difficulty
        self.current_scenario = None
        self._step_count = 0

    # ── reset ────────────────────────────────────────────────────────────────
    def reset(self) -> Observation:

        pool = [s for s in SCENARIOS if s["difficulty"] == self.difficulty]

        if not pool:
            pool = list(SCENARIOS)

        self.current_scenario = random.choice(pool)
        self._step_count = 0

        return Observation(
            video_context=self.current_scenario["video_context"],
            chat_history=self.current_scenario.get("chat_history", []),
            comment=self.current_scenario["comment"],
            difficulty=self.current_scenario["difficulty"],
            reward=0.0,
            done=False,
            message="Episode started. Moderate the comment.",
        )

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action: Action) -> StepResult:

        if self.current_scenario is None:
            raise RuntimeError("Call reset() before step()")

        self._step_count += 1

        # Difficulty-based grading
        if self.difficulty == "easy":
            score = grade_easy(
                prediction=action.decision,
                ground_truth=self.current_scenario["label"]
            )

        elif self.difficulty == "medium":
            score = grade_medium(
                prediction=action.decision,
                ground_truth=self.current_scenario["label"],
                context_match=True
            )

        else:
            score = grade_hard(
                prediction=action.decision,
                ground_truth=self.current_scenario["label"],
                justification=action.reason,
                threat_type=self.current_scenario.get("threat_type", "general")
            )

        # Multi-step episode settings
        max_steps_map = {
            "easy": 1,
            "medium": 3,
            "hard": 5
        }

        done = self._step_count >= max_steps_map[self.difficulty]

        info = {
            "correct_label": self.current_scenario["label"],
            "threat_type": self.current_scenario.get("threat_type", "general"),
            "step": self._step_count,
        }

        obs = Observation(
            video_context=self.current_scenario["video_context"],
            chat_history=self.current_scenario.get("chat_history", []),
            comment=self.current_scenario["comment"],
            difficulty=self.current_scenario["difficulty"],
            reward=float(score),
            done=done,
            message=f"Decision '{action.decision}' scored {score:.2f}",
        )

        # Load another scenario if episode continues
        if not done:
            pool = [s for s in SCENARIOS if s["difficulty"] == self.difficulty]

            if not pool:
                pool = list(SCENARIOS)

            self.current_scenario = random.choice(pool)

        return StepResult(
            score=float(score),
            reward=float(score),
            done=done,
            info=info,
            observation=obs,
        )

    # ── state ────────────────────────────────────────────────────────────────
    def state(self):

        if self.current_scenario is None:
            return {"status": "not_started"}

        return {
            "difficulty": self.difficulty,
            "step_count": self._step_count,
            "scenario": self.current_scenario,
        }