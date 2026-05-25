"""Adapter over the course-provided `millionaire_client` package.

The real game client, the course gives us (millionaire_client). Reimplement it we do NOT --
wrap it we do, and its Question/Option into our schemas.Question we translate.

Setup: sign up first in a browser at http://131.175.15.22:51111/ (a PoliMi email, one account).
On sys.path the `millionaire_client` package must be (on Colab, kept in Drive it is).
Polite we stay: rapid consecutive games/requests, avoid them we do.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from schemas import Question, QuestionType

# The option letters, in order these are (their server gives integer ids, not letters).
_LETTERS = "ABCDEFGH"

# A single "A) text" / "B. text" / "C: text" option line, this matches (the leading letter + its text).
_OPT_LINE = re.compile(r"^\s*([A-Za-z])\s*[).:\-]\s*(.+?)\s*$")


def _norm(s: str) -> str:
    # For comparison only: lowercased, whitespace collapsed, a trailing period dropped.
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


def strip_embedded_options(text: str, option_texts: list[str]) -> str:
    """A trailing 'A) ... B) ...' block, baked into the question text it sometimes is -- remove it we do.

    The danger (qid 617): the server's option ORDER differs from the text's embedded letters, so two
    conflicting A/B/C/D schemes the model would see. Strip the block we do -- but ONLY when its texts
    truly duplicate the server options (a subset match) AND the letters run A, B, C, ... in order.
    A question that merely mentions a letter, never mangle we will. Lossless this is -- the options,
    in EvalRecord.options preserved they already are.
    """
    if not text or not option_texts:
        return text
    lines = text.splitlines()
    while lines and not lines[-1].strip():  # Trailing blank lines, first drop them we do.
        lines.pop()

    letters: list[str] = []
    texts: list[str] = []
    n = len(lines)
    while n > 0:  # From the bottom up, the contiguous run of option-shaped lines we gather.
        m = _OPT_LINE.match(lines[n - 1])
        if not m:
            break
        letters.append(m.group(1).upper())
        texts.append(m.group(2))
        n -= 1
    letters.reverse()
    texts.reverse()

    if len(texts) < 2:  # Too short to be an options block, it is.
        return text
    # Consecutive A, B, C, ... the letters must be -- else a real list this may be, untouched leave it.
    if letters != [chr(ord("A") + i) for i in range(len(letters))]:
        return text
    # And a subset of the server options the texts must be -- only a true duplicate, strip we do.
    server = {_norm(t) for t in option_texts}
    if not {_norm(t) for t in texts} <= server:
        return text

    cleaned = "\n".join(lines[:n]).rstrip()
    return cleaned or text  # All text was the block? Then keep the original, defensive we stay.


def adapt_question(api_q) -> Question:
    """Their Question (options carry integer ids) -> our schemas.Question (with a letter->id map).

    The letter->id map, keep it we must -- by integer id the server wants the answer, not by letter.
    """
    options: dict[str, str] = {}
    option_ids: dict[str, int] = {}
    for i, opt in enumerate(api_q.options):
        letter = _LETTERS[i]
        options[letter] = opt.text
        option_ids[letter] = opt.id
    # A duplicated options block baked into the text, strip it we do -- the double-lettering, gone (qid 617).
    text = strip_embedded_options(api_q.text or "", list(options.values()))
    return Question(
        qid=str(api_q.id),
        text=text,
        options=options,
        option_ids=option_ids,
        qtype=QuestionType.MCQ,
        level=getattr(api_q, "level", None),
    )


class GameClient:
    """A thin wrapper over MillionaireClient. Login, list competitions, and play with OUR pipeline, it lets us.

    The provided package does the HTTP; the translation to/from our schemas, this class owns.
    """

    def __init__(self, base_url: str = "http://131.175.15.22:51111/", timeout: int = 30):
        try:
            from millionaire_client import MillionaireClient
        except ImportError as e:  # On sys.path the provided package must be, else fail loudly we do.
            raise ImportError(
                "The provided 'millionaire_client' package on sys.path it must be "
                "(folder NLP_assignment_api_client/). See README + PoliMillionaire.ipynb, you should."
            ) from e
        self._client = MillionaireClient(base_url, timeout=timeout)

    def login(self, username: str, password: str):
        # Authenticate we must, before competitions or games touch we can.
        return self._client.login(username, password)

    def list_competitions(self) -> list:
        # The competitions and their public ids (0,1,2,...), these it returns.
        return self._client.competitions.list_all()

    def play(
        self,
        competition_id: int,
        answer_fn: Callable[[Question], str],
        on_result: Optional[Callable] = None,
        mode: str = "text",
    ):
        """Play one full game with our pipeline as the strategy.

        Args:
            answer_fn: our Question -> a chosen letter ("A".."D"). The pipeline, this wraps.
            on_result: optional callback (our_question, letter, AnswerResult, time_remaining_s) -- for logging.
            mode: "text" or "speech".

        Note: against the 30s wall the answer_fn must stay -- seed its LatencyGuard from
        `game.time_remaining` minus a network margin, ideally you should.
        """
        game = self._client.game.start(competition_id, mode=mode)
        while game.in_progress:
            api_q = game.current_question
            if not api_q:  # No question left -- ended the game has.
                break
            q = adapt_question(api_q)
            time_left = game.time_remaining  # The server's truth on remaining seconds, this is.

            letter = answer_fn(q)  # Here, our pipeline decides.

            option_id = q.option_ids.get((letter or "").strip().upper())
            if option_id is None:
                # Parse failed -- but skip we cannot; guess the first option we must, an answer is owed.
                option_id = next(iter(q.option_ids.values()))

            result = game.answer(option_id)
            if on_result:
                on_result(q, letter, result, time_left)
            if result.game_over or result.timed_out:
                break
        return game
