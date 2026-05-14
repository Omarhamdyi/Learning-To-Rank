#!/usr/bin/env python3
"""Fast query typo correction based on corpus vocabulary and context."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence


def _char_ngrams(word: str, min_n: int = 2, max_n: int = 3) -> list[str]:
    grams: list[str] = []
    for n in range(min_n, max_n + 1):
        if len(word) < n:
            if n == min_n:
                grams.append(word)
            continue
        grams.extend(word[i : i + n] for i in range(len(word) - n + 1))
    return grams or [word]


def _bounded_damerau_levenshtein(a: str, b: str, max_distance: int) -> int:
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1

    prev = list(range(len(b) + 1))
    prev_prev: list[int] | None = None
    for i, ca in enumerate(a, start=1):
        curr = [i]
        min_in_row = curr[0]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            val = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
            if (
                i > 1
                and j > 1
                and prev_prev is not None
                and ca == b[j - 2]
                and a[i - 2] == cb
            ):
                # Adjacent transposition (Damerau-Levenshtein)
                val = min(val, prev_prev[j - 2] + 1)
            curr.append(val)
            if val < min_in_row:
                min_in_row = val
        if min_in_row > max_distance:
            return max_distance + 1
        prev_prev = prev
        prev = curr
    return prev[-1]


class FastTypoCorrector:
    def __init__(
        self,
        token_sequences: Iterable[Sequence[str]],
        context_sequences: Iterable[Sequence[str]] | None = None,
        min_token_freq: int = 3,
        min_ngram: int = 2,
        max_ngram: int = 3,
        max_candidates: int = 250,
    ) -> None:
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.max_candidates = max_candidates
        self.token_freq: Counter[str] = Counter()
        self.bigram_freq: Counter[tuple[str, str]] = Counter()
        self.ngram_index: dict[str, list[str]] = defaultdict(list)

        for seq in token_sequences:
            self.token_freq.update(seq)

        context_source = context_sequences if context_sequences is not None else token_sequences
        for seq in context_source:
            self.bigram_freq.update(zip(seq, seq[1:]))

        vocab = [
            tok
            for tok, freq in self.token_freq.items()
            if len(tok) >= 3 and freq >= min_token_freq and not tok.isdigit()
        ]
        self.vocab_set = set(vocab)

        for tok in vocab:
            for gram in set(_char_ngrams(tok, self.min_ngram, self.max_ngram)):
                self.ngram_index[gram].append(tok)

    def correct_token(self, token: str, prev_token: str | None = None) -> str:
        if token in self.vocab_set or len(token) < 3 or token.isdigit():
            return token

        overlap_counter: Counter[str] = Counter()
        for gram in set(_char_ngrams(token, self.min_ngram, self.max_ngram)):
            overlap_counter.update(self.ngram_index.get(gram, ()))
        if not overlap_counter:
            return token

        if len(token) <= 5:
            max_distance = 1
        elif len(token) <= 10:
            max_distance = 2
        else:
            max_distance = 3

        best_word = token
        best_distance = 99
        best_context = -1
        best_shape = -1
        best_overlap = -1
        best_key = (99, 0, 0, 0, 0)
        second_key: tuple[int, int, int, int, int] | None = None

        for cand, overlap in overlap_counter.most_common(self.max_candidates):
            if abs(len(cand) - len(token)) > max_distance:
                continue

            dist = _bounded_damerau_levenshtein(token, cand, max_distance=max_distance)
            if dist > max_distance:
                continue

            context_score = self.bigram_freq.get((prev_token, cand), 0) if prev_token else 0
            shape_score = (
                int(cand[0] == token[0])
                + int(cand[-1] == token[-1])
                + int(cand[:2] == token[:2])
                + int(cand[-2:] == token[-2:])
            )
            key = (
                dist,
                -context_score,
                -shape_score,
                -overlap,
                -self.token_freq[cand],
            )
            if key < best_key:
                second_key = best_key if best_word != token else None
                best_key = key
                best_word = cand
                best_distance = dist
                best_context = context_score
                best_shape = shape_score
                best_overlap = overlap
            elif second_key is None or key < second_key:
                second_key = key

        if best_word == token:
            return token

        # Distance-1 corrections are usually safe and recover most simple typos.
        if best_distance <= 1:
            return best_word

        # For distance-2+, require stronger lexical/context evidence.
        strong_signal = (
            best_context > 0
            or best_shape >= 2
            or best_overlap >= 3
            or self.token_freq[best_word] >= 8
        )
        if not strong_signal:
            return token

        # If runner-up is essentially tied, avoid aggressive rewrites.
        if second_key is not None and second_key[:2] == best_key[:2] and best_shape < 3:
            return token

        return best_word

    def correct_tokens(self, tokens: list[str]) -> list[str]:
        corrected: list[str] = []
        for token in tokens:
            prev = corrected[-1] if corrected else None
            corrected.append(self.correct_token(token, prev_token=prev))
        return corrected


_SEMANTIC_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "sofa",
        "sofas",
        "couch",
        "couches",
        "loveseat",
        "loveseats",
        "sectional",
        "sectionals",
        "settee",
        "settees",
        "futon",
        "futons",
    ),
    (
        "refrigerator",
        "refrigerators",
        "fridge",
        "fridges",
    ),
    (
        "faucet",
        "faucets",
        "tap",
        "taps",
        "spigot",
        "spigots",
    ),
    (
        "cabinet",
        "cabinets",
        "cupboard",
        "cupboards",
    ),
    (
        "mower",
        "mowers",
        "lawnmower",
        "lawnmowers",
    ),
    (
        "trimmer",
        "trimmers",
        "weedwacker",
        "weedwhacker",
        "weedeater",
        "stringtrimmer",
        "linetrimmer",
    ),
    (
        "flashlight",
        "flashlights",
        "torch",
        "torches",
    ),
    (
        "adhesive",
        "adhesives",
        "glue",
        "glues",
    ),
)

_CANONICAL_TOKEN_MAP: dict[str, str] = {}
_EQUIVALENT_TOKEN_MAP: dict[str, tuple[str, ...]] = {}
for _group in _SEMANTIC_SYNONYM_GROUPS:
    _unique = tuple(dict.fromkeys(_group))
    _canonical = _unique[0]
    for _tok in _unique:
        _CANONICAL_TOKEN_MAP[_tok] = _canonical
        _EQUIVALENT_TOKEN_MAP[_tok] = _unique


def canonicalize_semantic_tokens(tokens: list[str]) -> list[str]:
    return [_CANONICAL_TOKEN_MAP.get(tok, tok) for tok in tokens]


def expand_query_tokens_for_recall(tokens: list[str]) -> list[str]:
    """Expand canonical query tokens with tight synonym equivalents for recall.
    Keeps insertion order and de-duplicates to avoid unnecessary BM25 overhead."""
    expanded: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        equivalents = _EQUIVALENT_TOKEN_MAP.get(tok)
        if equivalents is None:
            equivalents = (_CANONICAL_TOKEN_MAP.get(tok, tok),)
        for cand in equivalents:
            if cand not in seen:
                seen.add(cand)
                expanded.append(cand)
    return expanded


def normalize_query_tokens(
    raw_query: str,
    normalize_text: callable,
    spelling_checker_dict: dict[str, str],
    typo_corrector: FastTypoCorrector | None,
) -> tuple[str, list[str]]:
    q_phrase = spelling_checker_dict.get(raw_query.lower().strip(), raw_query)
    q_norm = normalize_text(q_phrase)
    q_tok = q_norm.split()

    # If the phrase dictionary already corrected the query, keep that output.
    if q_phrase != raw_query:
        q_tok = canonicalize_semantic_tokens(q_tok)
        return " ".join(q_tok), q_tok

    dict_corrected_tokens: list[str] = []
    for tok in q_tok:
        repl = spelling_checker_dict.get(tok, tok)
        if " " in repl:
            repl = tok
        dict_corrected_tokens.append(repl)

    if typo_corrector is not None:
        typo_safe_tokens: list[str] = []
        for tok in dict_corrected_tokens:
            # Keep explicitly modeled semantic tokens stable (e.g., couch->sofa),
            # so typo correction doesn't rewrite them to unrelated words like "coach".
            if tok in _CANONICAL_TOKEN_MAP:
                typo_safe_tokens.append(tok)
            else:
                prev = typo_safe_tokens[-1] if typo_safe_tokens else None
                typo_safe_tokens.append(typo_corrector.correct_token(tok, prev_token=prev))
        dict_corrected_tokens = typo_safe_tokens

    dict_corrected_tokens = canonicalize_semantic_tokens(dict_corrected_tokens)
    q_norm = " ".join(dict_corrected_tokens)
    return q_norm, dict_corrected_tokens
