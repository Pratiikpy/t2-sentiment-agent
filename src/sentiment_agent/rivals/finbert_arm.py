"""The finBERT-weighted trader: ProsusAI/finBERT's sentence scores, traded per symbol.

finBERT (Araci, 2019; ProsusAI/finBERT, Apache-2.0) is BERT further trained on financial text and
fine-tuned on the Financial PhraseBank; it is the reference open model for financial sentiment and
the rival ARGUS's coordination-resistance result was measured against. Here it scores every sentence
of every post and headline naming a symbol, and the same pre-registered rule as the lexicon trader
(:class:`~sentiment_agent.rivals.registry.TextSentimentRule`) turns the per-symbol mean into a
weight, with the crowd (:data:`~sentiment_agent.rivals.registry.FINBERT_FOLLOW`) or against it
(:data:`~sentiment_agent.rivals.registry.FINBERT_FADE`).

What was read, and what was taken
---------------------------------
From ``ProsusAI/finBERT`` at commit ``44995e0c5870c4ab37a189d756550654ae87cdf0``:

* The sentence score is ``P(positive) - P(negative)`` after a softmax over the three classes
  (``finbert/finbert.py:624-625``). The class order is read from the model's own configuration
  (``id2label``), not assumed: it is ``{0: positive, 1: negative, 2: neutral}`` in the published
  weights, matching ``finbert/finbert.py:607-608``.
* Sentences are truncated to 64 tokens (``finbert/finbert.py:30``, ``:613``), in batches.
* Text is split into sentences first (``finbert/finbert.py:603``). Upstream uses NLTK's
  ``sent_tokenize``; this arm splits on sentence-ending punctuation and line breaks instead, to
  keep NLTK out of the dependencies. Posts are short and mostly one or two sentences, so the
  difference is small; it is a departure all the same, and is stated.

The model is loaded through ``transformers`` from the Hugging Face hub (``ProsusAI/finbert``,
pinned to revision :data:`FINBERT_REVISION`, the one read for this arm), only when the arm first
scores, and only if the optional ``[rivals]`` extra is installed. Nothing of finBERT is vendored.

The model's per-item score is the mean of its sentences' scores; an item with no sentence is not
scored. Each post counts once, copies included, as for the lexicon trader.
"""

import importlib
import importlib.util
import re
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Any, Final

from sentiment_agent.rivals.registry import (
    FINBERT_FADE,
    FINBERT_FOLLOW,
    Direction,
    TextSentimentRule,
    spec_for,
    texts_by_symbol,
    weights_from_scores,
)
from sentiment_agent.types import BookState, PerceptionSnapshot, Policy

FINBERT_MODEL_ID: Final = "ProsusAI/finbert"
FINBERT_REVISION: Final = "4556d13015211d73dccd3fdd39d39232506f3e43"
"""The Hugging Face revision of the weights read for this arm (``refs/main`` on 2026-09-24)."""
MAX_SEQ_LENGTH: Final = 64
"""``finbert/finbert.py:30``."""
LOOKBACK: Final = timedelta(hours=24)

SentenceScorer = Callable[[Sequence[str]], list[float]]
"""Scores sentences: ``P(positive) - P(negative)`` per sentence, in order."""


class FinbertUnavailable(RuntimeError):  # noqa: N818 - reads as the fact a failed run reports
    """The optional ``[rivals]`` extra (``transformers``, ``torch``) or the model weights are not
    available, so the finBERT arm cannot score."""


def finbert_installed() -> bool:
    """Whether the ``[rivals]`` extra is importable. The weights may still need a download."""
    return all(importlib.util.find_spec(name) is not None for name in ("transformers", "torch"))


_SENTENCE_END: Final = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Sentences of ``text``: split after ``.``, ``!`` or ``?`` followed by space, and at line
    breaks; blank pieces dropped, whitespace collapsed."""
    pieces = (" ".join(p.split()) for p in _SENTENCE_END.split(text))
    return [p for p in pieces if p]


class FinbertScorer:
    """ProsusAI/finbert, loaded on first use. Satisfies :data:`SentenceScorer`.

    ``local_files_only`` refuses any download (for offline runs and tests); ``batch_size`` bounds
    memory. Runs on CPU unless ``device`` says otherwise.
    """

    def __init__(
        self,
        *,
        model_id: str = FINBERT_MODEL_ID,
        revision: str = FINBERT_REVISION,
        local_files_only: bool = False,
        batch_size: int = 16,
        device: str = "cpu",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._model_id = model_id
        self._revision = revision
        self._local_only = local_files_only
        self._batch_size = batch_size
        self._device = device
        self._loaded: tuple[Any, Any, Any, int, int] | None = None

    def _load(self) -> tuple[Any, Any, Any, int, int]:
        if self._loaded is not None:
            return self._loaded
        if not finbert_installed():
            raise FinbertUnavailable(
                "the finBERT arm needs the optional [rivals] extra: pip install "
                "'t2-sentiment-agent[rivals]'"
            )
        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self._model_id, revision=self._revision, local_files_only=self._local_only
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self._model_id, revision=self._revision, local_files_only=self._local_only
            )
        except OSError as exc:
            raise FinbertUnavailable(
                f"could not load {self._model_id}@{self._revision}: {exc}"
            ) from exc
        model.to(self._device)
        model.eval()
        labels = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
        if "positive" not in labels or "negative" not in labels:
            raise FinbertUnavailable(
                f"{self._model_id} does not label its classes positive and negative: {labels}"
            )
        self._loaded = (torch, tokenizer, model, labels["positive"], labels["negative"])
        return self._loaded

    def __call__(self, sentences: Sequence[str]) -> list[float]:
        if not sentences:
            return []
        torch, tokenizer, model, positive, negative = self._load()
        scores: list[float] = []
        for start in range(0, len(sentences), self._batch_size):
            batch = list(sentences[start : start + self._batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                probabilities = torch.softmax(model(**encoded).logits, dim=-1)
            for row in probabilities.tolist():
                scores.append(float(row[positive]) - float(row[negative]))
        return scores


class FinbertArm:
    """The finBERT-weighted trader (module docstring).

    ``scorer`` replaces the model with any :data:`SentenceScorer`, for tests and for a scorer
    already loaded elsewhere; by default :class:`FinbertScorer` loads ``ProsusAI/finbert`` on first
    use. ``direction`` picks the registered arm.
    """

    MODEL_ID: Final = FINBERT_MODEL_ID

    def __init__(
        self,
        *,
        policy: Policy,
        direction: Direction = "follow",
        scorer: SentenceScorer | None = None,
        rule: TextSentimentRule | None = None,
        lookback: timedelta = LOOKBACK,
    ) -> None:
        self._rule = rule if rule is not None else TextSentimentRule(direction=direction)
        if self._rule.direction != direction:
            raise ValueError("the rule's direction must be the arm's")
        if lookback <= timedelta(0):
            raise ValueError("lookback must be positive")
        self.spec = spec_for(FINBERT_FOLLOW if direction == "follow" else FINBERT_FADE)
        self._policy = policy
        self._scorer: SentenceScorer = scorer if scorer is not None else FinbertScorer()
        self._lookback = lookback

    def score_texts(self, texts: Sequence[str]) -> list[float | None]:
        """Each text's mean sentence score, ``None`` for a text with no sentence. One scorer call
        for all of them, so batching spans texts."""
        sentences_per_text = [split_sentences(t) for t in texts]
        flat = [s for sentences in sentences_per_text for s in sentences]
        scored = self._scorer(flat)
        if len(scored) != len(flat):
            raise ValueError(f"the scorer returned {len(scored)} scores for {len(flat)} sentences")
        out: list[float | None] = []
        position = 0
        for sentences in sentences_per_text:
            n = len(sentences)
            chunk = scored[position : position + n]
            position += n
            out.append(sum(chunk) / n if n else None)
        return out

    def scores(self, snapshot: PerceptionSnapshot) -> dict[str, list[float]]:
        """Every scored text per symbol, most recent first."""
        texts = texts_by_symbol(snapshot, since=snapshot.taken_at - self._lookback)
        unique = {i.item_id: i.text for items in texts.values() for i in items}
        ids = list(unique)
        by_id = dict(zip(ids, self.score_texts([unique[i] for i in ids]), strict=True))
        result: dict[str, list[float]] = {}
        for symbol, items in texts.items():
            values = [v for i in items if (v := by_id[i.item_id]) is not None]
            if values:
                result[symbol] = values
        return result

    def targets(self, snapshot: PerceptionSnapshot, book: BookState) -> dict[str, float]:
        return weights_from_scores(
            self.scores(snapshot), book=book, rule=self._rule, policy=self._policy
        )


__all__ = [
    "FINBERT_MODEL_ID",
    "FINBERT_REVISION",
    "LOOKBACK",
    "MAX_SEQ_LENGTH",
    "FinbertArm",
    "FinbertScorer",
    "FinbertUnavailable",
    "SentenceScorer",
    "finbert_installed",
    "split_sentences",
]
