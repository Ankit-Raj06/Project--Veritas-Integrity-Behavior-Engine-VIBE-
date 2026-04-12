"""
VIBE – Veritas Integrity Behavior Engine
Inference script for Scaler/Meta PyTorch OpenEnv Hackathon.
Reads from environment variables:
    API_BASE_URL  – LLM router base URL
    MODEL_NAME    – model identifier
    HF_TOKEN      – Hugging Face token (used as API key)
Log format:
    [START] {...}
    [STEP]  {...}
    [END]   {...}
"""
import json
import os
import sys
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
API_KEY      = os.getenv("HF_TOKEN",     "")
DIFFICULTIES = ["easy", "medium", "hard"]
MAX_STEPS    = 10

# ── Score bounds (strictly open interval) ────────────────────────────────────
_SCORE_MIN: float = 0.001
_SCORE_MAX: float = 0.999

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are VIBE — a context-aware AI content moderation agent.
Your job: analyse a YouTube comment alongside its video context and decide how to moderate it.
You will receive:
- video_context: a description of the video the comment was posted on
- chat_history: previous comments in the same thread
- comment: the comment you must moderate
- difficulty: easy / medium / hard

DECISION OPTIONS:
- "safe"    → Comment is acceptable; no action needed
- "flag"    → Comment is borderline or context-dependent; escalate for human review
- "remove"  → Comment is clearly harmful, hateful, or violates policy

IMPORTANT RULES:
1. Context is everything. "bhai tu toh gaya 💀" on a gaming video = safe banter. Same phrase on a communal news video = flag/remove.
2. Hinglish and code-switching (Hindi+English mixed) is normal — do not flag purely based on non-English words.
3. Sarcasm, dark humour, and cultural idioms require context to judge correctly.
4. Consider the chat_history — a single comment may only be harmful in sequence.

Respond ONLY with a JSON object in this exact format (no extra text, no markdown):
{"decision": "<safe|flag|remove>", "reason": "<one sentence explanation>"}"""


# ── Score safety ─────────────────────────────────────────────────────────────
def safe_score(score) -> float:
    """Clamp score to strictly open interval (0, 1) using named constants."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0.05
    return min(max(s, _SCORE_MIN), _SCORE_MAX)


# ── Helpers ───────────────────────────────────────────────────────────────────
def log(tag: str, data: dict) -> None:
    """Print a log line in the exact format the Scaler validator reads."""
    print(f"[{tag}] {json.dumps(data)}", flush=True)


def call_llm(client: OpenAI, observation: dict) -> dict:
    """Call the LLM and return parsed {decision, reason}."""
    user_message = (
        f"Video context: {observation.get('video_context', '')}\n"
        f"Chat history: {json.dumps(observation.get('chat_history', []))}\n"
        f"Comment to moderate: {observation.get('comment', '')}\n"
        f"Difficulty: {observation.get('difficulty', 'easy')}"
    )
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.1,
                max_tokens=150,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            if parsed.get("decision") in ("safe", "flag", "remove"):
                return parsed
        except Exception as e:
            if attempt == 2:
                return {"decision": "flag", "reason": f"LLM error after retries: {e}"}
    return {"decision": "flag", "reason": "Could not parse LLM response"}


def run_episode(client: OpenAI, env, difficulty: str, episode_num: int) -> dict:
    """Run one full episode and return result dict."""
    task_name = f"task_{difficulty}"
    log("START", {
        "task":       task_name,
        "episode":    episode_num,
        "difficulty": difficulty,
    })

    # Reset environment
    obs_obj = env.reset()
    if hasattr(obs_obj, "model_dump"):
        obs = obs_obj.model_dump()
    elif hasattr(obs_obj, "dict"):
        obs = obs_obj.dict()
    else:
        obs = dict(obs_obj) if obs_obj else {}

    total_reward = 0.0
    step_num     = 0

    for step_num in range(1, MAX_STEPS + 1):
        # Get action from LLM
        action_dict = call_llm(client, obs)

        # Build Action object
        try:
            from environment.env import Action
        except ImportError:
            from env import Action

        action = Action(
            decision=action_dict["decision"],
            reason=action_dict.get("reason", ""),
        )

        # Step environment
        result = env.step(action)

        # Normalise result
        if isinstance(result, tuple):
            score, done, info = result
        elif isinstance(result, dict):
            score = result.get("score", result.get("reward", 0.05))
            done  = result.get("done", True)
            info  = result.get("info", {})
        else:
            score = getattr(result, "reward", getattr(result, "score", 0.05))
            done  = getattr(result, "done", True)
            info  = getattr(result, "info", {})

        # Clamp step score before logging
        step_score = safe_score(score)
        total_reward += step_score

        log("STEP", {
            "step":     step_num,
            "decision": action_dict["decision"],
            "reason":   action_dict.get("reason", ""),
            "reward":   step_score,
            "correct":  info.get("correct_label", "unknown") if isinstance(info, dict) else "unknown",
            "done":     bool(done),
        })

        if done:
            break

    # Clamp final score before logging
    final_score = safe_score(total_reward / max(step_num, 1))

    log("END", {
        "task":         task_name,
        "episode":      episode_num,
        "total_reward": round(total_reward, 4),
        "steps":        step_num,
        "score":        final_score,
    })

    return {
        "task":         task_name,
        "score":        final_score,
        "total_reward": total_reward,
        "steps_taken":  step_num,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    try:
        from environment.env import AISafetyEnv
    except ImportError:
        try:
            from env import AISafetyEnv
        except ImportError:
            print(
                "[ERROR] Cannot import AISafetyEnv. "
                "Make sure environment/env.py exists with class AISafetyEnv.",
                file=sys.stderr,
            )
            sys.exit(1)

    all_results = []
    for i, difficulty in enumerate(DIFFICULTIES):
        env = AISafetyEnv(difficulty=difficulty)
        result = run_episode(client, env, difficulty, episode_num=i + 1)
        all_results.append(result)

    # Clamp overall score
    overall = safe_score(sum(r["score"] for r in all_results) / len(all_results))

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['task']:15s} | score: {r['score']:.4f} | steps: {r['steps_taken']}")
    print(f"\n  Overall score: {overall:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
