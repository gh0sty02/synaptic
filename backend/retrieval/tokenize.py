import re

_STOPWORDS = frozenset(
    {
        "the",
        "is",
        "a",
        "an",
        "of",
        "to",
        "in",
        "for",
        "and",
        "or",
        "on",
        "with",
        "this",
        "that",
        "it",
        "be",
        "as",
        "are",
        "was",
        "were",
    }
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        for part in raw.split("_"):  # snake_case -> sub-parts
            if not part:
                continue
            for sub in _CAMEL_BOUNDARY_RE.sub(" ", part).split() or [
                part
            ]:  # camelCase -> sub-parts
                lowered = sub.lower()
                if lowered and lowered not in _STOPWORDS:
                    tokens.append(lowered)
    return tokens
