from __future__ import annotations

import re

import pandas as pd

TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""
    return " ".join(TOKEN_RE.findall(str(text).lower()))


def get_char_ngrams(text: str, n: int = 3) -> list[str]:
    tokens: list[str] = []
    for word in text.split():
        if len(word) < n:
            tokens.append(word)
        else:
            tokens.extend([word[i : i + n] for i in range(len(word) - n + 1)])
    return tokens

