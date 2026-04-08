"""
VIBE – Dataset
Loads scenarios from the real hate-speech CSV.
Falls back to synthetic scenarios if the CSV is not found.

CSV columns expected:
    context      – video description / caption
    comment      – the comment to moderate
    final_label  – 0 = safe, 1 = toxic/remove
    toxic_flag   – True = profane but context-dependent → flag
"""

import os
import random
import csv
from typing import List, Dict, Any

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hate_speech_data.csv")


# ── CSV loader ────────────────────────────────────────────────────────────────

def _load_from_csv(path: str) -> List[Dict[str, Any]]:
    scenarios = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                context     = str(row.get("context", "")).strip()
                comment     = str(row.get("comment", "")).strip()
                final_label = str(row.get("final_label", "0")).strip()
                toxic_flag  = str(row.get("toxic_flag", "False")).strip().lower() == "true"

                if not context or not comment:
                    continue

                # Derive label
                if toxic_flag:
                    label      = "flag"
                    difficulty = "medium"
                elif final_label == "1":
                    label      = "remove"
                    difficulty = "easy" if i % 3 != 0 else "hard"
                else:
                    label      = "safe"
                    difficulty = "easy"

                # Hard scenarios: ambiguous context
                if i % 10 == 0 and label == "remove":
                    difficulty = "hard"

                scenarios.append({
                    "video_context": context,
                    "chat_history":  [],
                    "comment":       comment,
                    "label":         label,
                    "difficulty":    difficulty,
                    "threat_type":   "hate_speech" if label == "remove" else "general",
                })
    except FileNotFoundError:
        pass  # handled below
    return scenarios


# ── Synthetic fallback ────────────────────────────────────────────────────────

SYNTHETIC_SCENARIOS: List[Dict[str, Any]] = [
    # ── Easy ──
    {
        "video_context":  "Minecraft speedrun fails compilation — funny moments",
        "chat_history":   [],
        "comment":        "bhai tu toh gaya 💀💀 absolute trash player lmao",
        "label":          "safe",
        "difficulty":     "easy",
        "threat_type":    "general",
    },
    {
        "video_context":  "Political rally highlights — communal tensions discussion",
        "chat_history":   [],
        "comment":        "These people should be thrown out of the country",
        "label":          "remove",
        "difficulty":     "easy",
        "threat_type":    "hate_speech",
    },
    {
        "video_context":  "Cooking tutorial: how to make biryani at home",
        "chat_history":   [],
        "comment":        "This recipe is absolutely cooked 😂 love it!",
        "label":          "safe",
        "difficulty":     "easy",
        "threat_type":    "general",
    },
    {
        "video_context":  "Stand-up comedy special — dark humour",
        "chat_history":   [],
        "comment":        "This comedian is a disgrace, shouldn't be allowed on stage",
        "label":          "flag",
        "difficulty":     "easy",
        "threat_type":    "harassment",
    },
    # ── Medium ──
    {
        "video_context":  "News report: religious violence incidents in various states",
        "chat_history":   [
            {"user": "user_a", "comment": "This is getting out of hand"},
            {"user": "user_b", "comment": "One community is always behind this"},
        ],
        "comment":        "We all know who's responsible. Time to take action.",
        "label":          "flag",
        "difficulty":     "medium",
        "threat_type":    "incitement",
    },
    {
        "video_context":  "FIFA 24 gameplay — competitive ranked match",
        "chat_history":   [
            {"user": "user_a", "comment": "gg wp both teams"},
            {"user": "user_b", "comment": "nice save bro"},
        ],
        "comment":        "you're absolutely cooked, skill diff 💀",
        "label":          "safe",
        "difficulty":     "medium",
        "threat_type":    "general",
    },
    {
        "video_context":  "Documentary on caste discrimination in rural India",
        "chat_history":   [],
        "comment":        "yaar these people will never change, it's in their DNA",
        "label":          "remove",
        "difficulty":     "medium",
        "threat_type":    "hate_speech",
    },
    {
        "video_context":  "Bollywood movie trailer — romantic film",
        "chat_history":   [
            {"user": "user_a", "comment": "Can't wait for this 🔥"},
        ],
        "comment":        "The actress looks so ugly, who cast her 🤮",
        "label":          "flag",
        "difficulty":     "medium",
        "threat_type":    "harassment",
    },
    # ── Hard ──
    {
        "video_context":  "Comedy roast battle — celebrities roasting each other",
        "chat_history":   [
            {"user": "host",   "comment": "Alright, gloves off tonight everyone!"},
            {"user": "user_a", "comment": "This is brutal 😂"},
        ],
        "comment":        "They should just cancel that talentless joke of a person for good",
        "label":          "flag",
        "difficulty":     "hard",
        "threat_type":    "ambiguous",
    },
    {
        "video_context":  "University debate on freedom of speech vs hate speech laws",
        "chat_history":   [
            {"user": "user_a", "comment": "Both sides make valid points"},
        ],
        "comment":        "People who defend hate speech should face consequences themselves",
        "label":          "flag",
        "difficulty":     "hard",
        "threat_type":    "ambiguous",
    },
    {
        "video_context":  "Cricket match highlights — India vs Pakistan",
        "chat_history":   [
            {"user": "user_a", "comment": "India played well tonight"},
            {"user": "user_b", "comment": "Pakistan couldn't handle the pressure 😬"},
        ],
        "comment":        "Pakistanis are all the same, can't win even when given a chance 🙄",
        "label":          "remove",
        "difficulty":     "hard",
        "threat_type":    "hate_speech",
    },
    {
        "video_context":  "Reaction video to a viral meme — humor channel",
        "chat_history":   [],
        "comment":        "main toh mar jaaunga yaar this is too funny 💀😂",
        "label":          "safe",
        "difficulty":     "hard",
        "threat_type":    "general",
    },
]


# ── Public dataset ─────────────────────────────────────────────────────────────

def _build_scenarios() -> List[Dict[str, Any]]:
    csv_data = _load_from_csv(CSV_PATH)
    if len(csv_data) >= 20:
        # Use real data but always include at least 4 synthetic hard scenarios
        # (CSV may have fewer hard/ambiguous examples)
        hard_synthetic = [s for s in SYNTHETIC_SCENARIOS if s["difficulty"] == "hard"]
        return csv_data + hard_synthetic
    # Not enough real data — use synthetic
    return SYNTHETIC_SCENARIOS


SCENARIOS: List[Dict[str, Any]] = _build_scenarios()


# ── Convenience getters ───────────────────────────────────────────────────────

def get_scenario(difficulty: str = "easy") -> Dict[str, Any]:
    pool = [s for s in SCENARIOS if s["difficulty"] == difficulty]
    if not pool:
        pool = SCENARIOS
    return random.choice(pool)


def get_task1_scenario() -> Dict[str, Any]:
    return get_scenario("easy")


def get_task2_scenario() -> Dict[str, Any]:
    return get_scenario("medium")


def get_task3_scenario() -> Dict[str, Any]:
    return get_scenario("hard")
