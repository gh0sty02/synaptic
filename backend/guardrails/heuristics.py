import re

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"ignore (all|any|the) (previous|prior|above) instructions", re.I),
        "injection",
    ),
    (
        re.compile(
            r"disregard (your|the) (system prompt|instructions|guidelines)", re.I
        ),
        "injection",
    ),
    (re.compile(r"you are (now )?DAN\b", re.I), "injection"),
    (re.compile(r"\bdo anything now\b", re.I), "injection"),
    (
        re.compile(r"no (content policy|restrictions|filters) (anymore|now)?", re.I),
        "injection",
    ),
    (re.compile(r"repeat (the text|everything) above", re.I), "injection"),
    (re.compile(r"reveal your (system prompt|instructions)", re.I), "injection"),
    (
        re.compile(
            r"pretend (you are|to be) .*(unrestricted|jailbroken|no rules)", re.I
        ),
        "injection",
    ),
]


def check(query: str) -> str | None:
    """Cheap regex/keyword pre-filter. Returns a category name on a hit, None otherwise.

    Deliberately narrow — exists to catch well-known jailbreak phrasing at zero cost,
    not to be the primary defense. Coverage grows from plan-guardrails.md item 6's
    labeled test set false negatives, not from guessing more patterns up front.
    """
    for pattern, category in _PATTERNS:
        if pattern.search(query):
            return category
    return None
