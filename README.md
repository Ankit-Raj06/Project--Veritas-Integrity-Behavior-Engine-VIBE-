# VIBE – Veritas Integrity Behavior Engine

> **Context-aware AI content moderation OpenEnv environment**
> Built for the Meta PyTorch OpenEnv Hackathon @ Scaler School of Technology

---

## What Is VIBE?

VIBE is a real-world OpenEnv environment where an AI agent moderates YouTube comments by reasoning about **context** — not just the comment text.

The core insight: the same comment means entirely different things depending on the video it's posted on.

> `"bhai tu toh gaya 💀"` on a Minecraft fails video = **safe banter**  
> `"bhai tu toh gaya 💀"` on a communal tensions news video = **escalate/remove**

Most moderation systems fail at this. VIBE tests whether an agent can reason about video context, chat thread history, cultural tone, and Hinglish/code-switching patterns.

---

## Environment Description

### Action Space

| Field | Type | Values |
|-------|------|--------|
| `decision` | string | `safe` \| `flag` \| `remove` |
| `reason` | string | Agent's free-text justification |

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `video_context` | string | Video description (thumbnail caption or transcript snippet) |
| `chat_history` | list | Previous comments in the same thread |
| `comment` | string | The comment to moderate |
| `difficulty` | string | `easy` / `medium` / `hard` |
| `reward` | float | Reward from the previous step (0.0 on reset) |
| `done` | bool | Whether the episode has ended |
| `message` | string | Human-readable feedback on the last action |

### Three Tasks

| Task | Difficulty | What the agent does | Grader |
|------|-----------|---------------------|--------|
| `task_easy` | Easy | Moderate a single comment with clear label | Exact match grader |
| `task_medium` | Medium | Moderate a comment inside a reply thread | Thread-context grader (+0.1 bonus for consistency) |
| `task_hard` | Hard | Moderate ambiguous/Hinglish/multi-cultural comments | Full grader (+0.15 bonus for quality justification) |

### Reward Function

```
Exact match (pred == truth)           → 1.0
Predicted "flag", truth is "remove"   → 0.5   (cautious, partial credit)
Predicted "flag", truth is "safe"     → 0.4   (over-cautious)
Predicted "remove", truth is "safe"   → 0.1   (false positive, penalised)
Predicted "safe", truth is "remove"   → 0.0   (missed harmful content)
```

Partial rewards are given at every step — no sparse reward problem.

---

## Setup Instructions

### Prerequisites

- Python 3.11
- Docker Desktop
- A Hugging Face account with a **Write** token

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/vibe-openenv.git
cd vibe-openenv

conda create -n openenv-env python=3.11
conda activate openenv-env

pip install -r requirements.txt
pip install openenv-core
```

### 2. Set environment variables

```bash
# Windows (Anaconda Prompt)
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
set HF_TOKEN=hf_your_token_here

# Mac/Linux
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=hf_your_token_here
```

### 3. Run the server locally

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### 4. Run inference

```bash
python inference.py
```

Expected output:
```
[START] {"task": "task_easy", "episode": 1, "difficulty": "easy"}
[STEP]  {"step": 1, "decision": "remove", "reward": 1.0, "done": true}
[END]   {"task": "task_easy", "total_reward": 1.0, "score": 1.0}
```

### 5. Run with Docker

```bash
docker build . -t vibe-env:latest
docker run -p 7860:7860 vibe-env:latest
```

### 6. Validate submission

```bash
openenv validate
```

---

## Dataset

VIBE uses a real hate speech dataset (`data/hate_speech_data.csv`) with 10,000 labelled YouTube comments and their video context captions.

| Column | Description |
|--------|-------------|
| `context` | Video description / caption |
| `comment` | The comment text |
| `final_label` | 0 = safe, 1 = toxic |
| `toxic_flag` | True = profane but context-dependent (→ flag) |

If the CSV is absent, the environment falls back to 12 built-in synthetic scenarios covering all difficulty levels.

---

## Baseline Scores

| Task | Difficulty | Baseline Score |
|------|-----------|---------------|
| task_easy | Easy | 0.72 |
| task_medium | Medium | 0.61 |
| task_hard | Hard | 0.48 |
| **Overall** | | **0.60** |

---

## File Structure

```
vibe-openenv/
├── inference.py          ← Agent inference script (Person 3)
├── app.py                ← FastAPI server with /reset /step /state
├── openenv.yaml          ← OpenEnv spec file
├── Dockerfile            ← Root-level, builds correctly for validator
├── requirements.txt
├── README.md
├── data/
│   └── hate_speech_data.csv
└── environment/
    ├── __init__.py
    ├── env.py            ← AISafetyEnv with reset() / step() / state()
    ├── dataset.py        ← Real CSV loader + synthetic fallback
    └── graders.py        ← Scoring functions for all 3 tasks
```
