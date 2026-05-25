# API & Interface Contracts

> The authoritative list of interfaces between modules. Keep this in sync with `src/`.
> If you change a signature in code, change it here in the SAME session. No drift.

---

## A. Internal data contract (`src/schemas.py`) — single source of truth

```python
class QuestionType(str, Enum): MCQ="mcq"; OPEN="open"; UNKNOWN="unknown"

@dataclass
class Question:
    qid: str
    text: str
    options: dict[str,str]   # {"A": "...", ...}; empty for open questions
    option_ids: dict[str,int] # {"A": 101, ...}; the SERVER's Option.id per letter (for submitting)
    qtype: QuestionType = MCQ
    level: int | None        # difficulty rung 1..15 (game-provided)
    topic: str | None        # filled by classifier
    language: str | None     # "en"/"it"
    gold: str | None         # truth; only for dev sets, None in live play

@dataclass
class RetrievedDoc:
    doc_id: str; text: str; source: str; score: float

@dataclass
class Prediction:
    qid: str; answer: str; confidence: float; raw_output: str
    model: str; prompt_strategy: str
    retrieval_used: bool; retrieved_doc_ids: list[str]
    tool_used: str | None; latency_s: float
    tokens_in: int; tokens_out: int; error: str | None

@dataclass
class EvalRecord:    # one JSONL row; mirrors the rubric's required fields
    run_id; timestamp; qid; question_text; qtype; topic; level; language
    model; prompt_strategy; retrieval_used; retrieved_doc_ids; tool_used
    predicted_answer; gold_answer; correct (bool|None); confidence
    latency_s; latency_breakdown(dict); tokens_in; tokens_out; raw_output; error
```

## B. Module interfaces (stable signatures)

```python
# inference/engine.py
class LLMEngine(ABC):
    def generate(self, prompt: str, max_new_tokens=256, temperature=0.0) -> str
    def warmup(self) -> None

# prompting/builder.py
class PromptBuilder:
    def __init__(self, strategy: str = "zero_shot_v1")
    def build(self, question: Question, context: list[RetrievedDoc] | None = None) -> str

# classify/classifier.py
class QuestionClassifier:
    def classify(self, question: Question) -> Question      # returns enriched copy
    def needs_calculator(self, question: Question) -> bool
    def needs_retrieval(self, question: Question) -> bool

# retrieval/retriever.py
class Retriever:
    def __init__(self, index_path: str, embedder: str, top_k: int = 3)
    def retrieve(self, question: Question) -> list[RetrievedDoc]

# tools/calculator.py
def calculate(expression: str) -> float

# agent/pipeline.py
class QAPipeline:
    def __init__(self, engine, prompt_builder, classifier=None, retriever=None,
                 tools=None, latency_budget_s=30.0)
    def answer(self, question: Question) -> Prediction       # THE seam
    @staticmethod
    def parse_answer(raw_output: str, question: Question) -> tuple[str, float]

# agent/voting.py
def majority_vote(predictions: list[Prediction]) -> Prediction

# evaluation/logger.py
class ExperimentLogger:
    def __init__(self, root: str, run_id: str, meta: dict | None = None)
    def log(self, record: EvalRecord) -> None
    def close(self) -> None

# evaluation/runner.py  -- TWO MODES, one EvalRecord/JSONL format (D-015)
class BenchmarkRunner:        # OFFLINE ("our own test"): dev set, correct from gold
    def __init__(self, pipeline, config: RunConfig, log_root="experiments/runs")
    def run(self, questions: list[Question]) -> str          # returns run path
class LiveRunner:             # LIVE ("real test"): game API drives loop, correct from AnswerResult
    def __init__(self, pipeline, config: RunConfig, game_client, log_root="experiments/runs")
    def run(self) -> str                                     # returns run path
def run_session(pipeline, config, *, questions=None, game_client=None,
                log_root="experiments/runs") -> str          # THE switch: dispatches on config.mode
```

## C. EXTERNAL: Game API — ✅ CONFIRMED (provided `millionaire_client` package)

**Source of truth:** `NLP_assignment_api_client/millionaire_client/` (+ `PoliMillionaire.ipynb` demo).
We **wrap** it (do NOT reimplement) via `src/game/client.py::GameClient` (D-014).

**Setup:** sign up in a browser at `http://131.175.15.22:51111/` (PoliMi email, 1 account/email).
Keep `millionaire_client` on `sys.path` (Drive on Colab). `API_URL = "http://131.175.15.22:51111/"`.

**Competitions (confirmed LIVE 2026-05-25 via smoke test; all `max_levels=15`):**
| id | name |
|---|---|
| 0 | Entertainment |
| 1 | Ancient History and Politics |
| 2 | Science and Nature |
| 3 | Maths |
| 4 | Philosophy and Psychology |
| 5 | News |

Topic-driven routing falls out of these: **id 3 (Maths) → calculator tool** (Phase 3 ablation target);
**id 5 (News) → RAG** (current events beat the model's knowledge cutoff → Phase 4 target). The six names
double as the fixed `topic` label set for the per-topic accuracy breakdown (rubric "per-topic strengths").

**Provided API (the parts we use):**
```python
from millionaire_client import MillionaireClient, AuthenticationError
client = MillionaireClient(API_URL, timeout=30)
client.login(username, password)                       # cookie 'polimillionaire_auth'
comps = client.competitions.list_all()                 # [Competition(id, name, max_levels=15, ...)] ; id = 0,1,2,...
game = client.game.start(competition_id, mode="text")  # mode in {"text","speech"} -> GameSession
while game.in_progress:
    q = game.current_question      # models.Question(id:int, text:str, options:[Option(id:int, text:str)], level:int)
    t = game.time_remaining        # seconds to deadline (server truth); game.state.question_deadline
    result = game.answer(option_id)   # SUBMIT BY INTEGER Option.id  (or game.answer_by_text(text))
    # result: AnswerResult(correct:bool|None, game_over:bool, earned_amount:float, timed_out:bool,
    #         status, current_level, reached_level, question=<NEXT question>, money_pyramid)
    # game auto-advances internal state to the next question.
client.leaderboard.get(competition_id, limit=10, mode="text")  # separate "text"/"speech" boards
client.play_game(competition_id, answer_strategy, mode)        # convenience loop; strategy: Question -> id|text
```

**Critical facts:**
- **MCQ, answer by integer `Option.id`** (not a letter). Our pipeline emits a letter → map via
  `Question.option_ids[letter]` → submit. `adapt_question()` builds that map.
- **30s/question, NO timeout push.** Must still submit after 30s; late submit (even if correct) →
  `timed_out`. Network RTT to the server counts → aim to answer in **~25s**. Seed `LatencyGuard` from
  `game.time_remaining`.
- **Levels 1..15**, increasing difficulty; `money_pyramid`, `earned_amount`, safe amounts.
- **`correct` is only known AFTER submitting** → fill `EvalRecord.correct` from `AnswerResult.correct`
  in live play (offline accuracy needs our own dev set).
- **Speech mode is LIVE:** `game.start(mode="speech")`, `game.fetch_audio_question()`,
  `game.fetch_audio_option_next()` (sequential A→D), returns WAV bytes. **Timer starts when option D is
  fetched.** Enables a Whisper-STT audio path (Session 11) as a real bonus, not "future".
- **Rate limiting:** server raises `RateLimitError` (429) → be polite, no hot loops.

**Our wrapper (`src/game/client.py`):**
```python
adapt_question(api_q) -> schemas.Question        # maps Option list -> options{} + option_ids{}
class GameClient:
    def __init__(base_url=API_URL, timeout=30)
    def login(username, password)
    def list_competitions() -> list
    def play(competition_id, answer_fn: Question->letter, on_result=None, mode="text")  # the pipeline bridge

## D. Config contract (`config.py`)

`RunConfig`: `run_id, seed, latency_budget_s, prompt_strategy, mode, dataset_path,
model(ModelConfig), retrieval(RetrievalConfig), game(GameConfig), extra`. Loaded from
`configs/*.yaml`, serialized into each run's `meta.json` (`meta["mode"]` always recorded).
- `mode`: `"offline"` (our own dev-set test) | `"live"` (real game API). Free-text aliases
  accepted via `schemas.RunMode.normalize` ("our own test"→offline, "real test"→live).
- `GameConfig`: `competition_id(0..5), game_mode("text"|"speech"), aim_seconds(=25.0, the
  network margin below the 30s wall)`. Ignored in offline mode.
- Configs: `configs/base.yaml` (mode=offline default) and `configs/live.yaml` (mode=live).

## E. Experiment log layout (filesystem contract)

```
experiments/runs/<run_id>/
├── records.jsonl   # one EvalRecord per line (flushed per write)
└── meta.json       # full RunConfig + model + git commit + hardware + seed
```
