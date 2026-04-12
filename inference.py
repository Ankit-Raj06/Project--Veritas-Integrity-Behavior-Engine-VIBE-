#!/usr/bin/env python3
"""
VIBE – Veritas Integrity Behavior Engine
Inference script for Scaler/Meta PyTorch OpenEnv Hackathon.

Reads from environment variables:
  API_BASE_URL – LLM router base URL
  MODEL_NAME   – model identifier
  HF_TOKEN     – Hugging Face token (used as API key)
"""

import os
import json
import time
import logging
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

from graders import grade_easy, grade_medium, grade_hard

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Environment ───────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "EMPTY")


# ── Safety clamp (mirrors graders.py) ────────────────────────────────────────
def safe_score(score: float) -> float:
    """Ensure score is strictly within (0, 1) — never exactly 0.0 or 1.0."""
    return max(0.001, min(0.999, float(score)))


# ── LLM call ─────────────────────────────────────────────────────────────────
def call_llm(
    system_prompt: str,
    user_prompt:   str,
    temperature:   float = 0.0,
    max_tokens:    int   = 256,
) -> str:
    """Call the LLM and return the response text. Returns empty string on error."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return ""


# ── Prompt templates ──────────────────────────────────────────────────────────
SYSTEM_EASY = """You are a content moderation classifier.
Classify the following comment as one of: safe, flag, remove.
Respond with ONLY one word — the label."""

SYSTEM_MEDIUM = """You are a content moderation classifier reviewing a discussion thread.
Classify the final comment in context as one of: safe, flag, remove.
Respond with ONLY one word — the label."""

SYSTEM_HARD = """You are a senior content moderation analyst.
You will be given a post (possibly cross-platform or multi-modal).
Return a JSON object with exactly two keys:
  "decision": one of "safe", "flag", or "remove"
  "justification": a concise explanation (≥10 words) citing specific signals you observed.
Respond with ONLY valid JSON, no markdown fences."""


# ── Task runners ──────────────────────────────────────────────────────────────
def run_easy(obs: Dict[str, Any]) -> Tuple[str, float]:
    """Task 1 — single comment, unambiguous label."""
    comment = obs.get("comment", obs.get("text", ""))
    prediction = call_llm(SYSTEM_EASY, comment) or "flag"
    score = grade_easy(prediction, obs.get("label", obs.get("ground_truth", "flag")))
    return prediction, safe_score(score)


def run_medium(obs: Dict[str, Any]) -> Tuple[str, float]:
    """Task 2 — thread moderation with context."""
    thread  = obs.get("thread", [])
    comment = obs.get("comment", obs.get("text", ""))

    context_lines = "\n".join(
        f"[{i+1}] {c}" for i, c in enumerate(thread)
    )
    user_prompt = f"Thread context:\n{context_lines}\n\nComment to classify:\n{comment}"

    prediction    = call_llm(SYSTEM_MEDIUM, user_prompt) or "flag"
    context_match = obs.get("context_match", True)
    score = grade_medium(
        prediction,
        obs.get("label", obs.get("ground_truth", "flag")),
        context_match=context_match,
    )
    return prediction, safe_score(score)


def run_hard(obs: Dict[str, Any]) -> Tuple[str, float]:
    """Task 3 — cross-platform / multi-modal with justification."""
    content = obs.get("content", obs.get("text", obs.get("comment", "")))
    threat_type = obs.get("threat_type", "general")

    raw = call_llm(SYSTEM_HARD, content, max_tokens=512) or "{}"

    # Strip accidental markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])

    try:
        parsed = json.loads(raw)
        prediction    = parsed.get("decision", "flag")
        justification = parsed.get("justification", "")
    except json.JSONDecodeError:
        logger.warning("Hard task JSON parse failed; raw=%r", raw)
        prediction    = "flag"
        justification = ""

    score = grade_hard(
        prediction,
        obs.get("label", obs.get("ground_truth", "flag")),
        justification=justification,
        threat_type=threat_type,
    )
    return prediction, safe_score(score)


# ── Episode runner ────────────────────────────────────────────────────────────
def run_episode(env) -> float:
    """
    Drive one episode of the VIBE environment.
    Returns the final score strictly in (0, 1).
    """
    obs   = env.reset()
    total_reward = 0.0
    step_num     = 0
    done         = False

    while not done:
        step_num += 1

        # Determine task difficulty
        difficulty = obs.get("difficulty", "easy") if isinstance(obs, dict) else "easy"

        if difficulty == "hard":
            action, step_score = run_hard(obs)
        elif difficulty == "medium":
            action, step_score = run_medium(obs)
        else:
            action, step_score = run_easy(obs)

        # Submit action to environment
        result = env.step(action)

        # Parse env response — handle tuple, dict, or object
        if isinstance(result, tuple):
            env_score, done, info = result[0], result[1], result[2] if len(result) > 2 else {}
            env_score = safe_score(env_score)
        elif isinstance(result, dict):
            env_score = safe_score(result.get("score", result.get("reward", 0.1)))
            done      = result.get("done", True)
            info      = result.get("info", {})
        else:
            env_score = safe_score(getattr(result, "reward", getattr(result, "score", 0.1)))
            done      = getattr(result, "done", True)
            info      = getattr(result, "info", {})

        # Use our grader score when the env doesn't return a meaningful one
        combined_score = safe_score((step_score + env_score) / 2.0)
        total_reward  += combined_score

        logger.info(
            "step=%d difficulty=%s action=%s grader=%.4f env=%.4f combined=%.4f",
            step_num, difficulty, action, step_score, env_score, combined_score,
        )

        # Advance observation
        if isinstance(result, tuple) and len(result) > 2:
            obs = info.get("next_obs", obs)
        elif isinstance(result, dict):
            obs = result.get("obs", result.get("next_obs", obs))
        else:
            obs = getattr(result, "obs", getattr(result, "next_obs", obs))

    final_score = safe_score(total_reward / max(step_num, 1))
    logger.info("Episode finished: steps=%d final_score=%.4f", step_num, final_score)
    return final_score


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    try:
        from environment import VIBEEnvironment  # adjust import to your env module
        env = VIBEEnvironment()
    except ImportError:
        logger.error(
            "Could not import VIBEEnvironment. "
            "Make sure 'environment.py' is on PYTHONPATH."
        )
        raise

    score = run_episode(env)
    print(f"Final score: {score:.4f}")


if __name__ == "__main__":
    main()
