import re


SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9_-]+",          # API-key-like value
    r"api[_-]?key\s*[:=]\s*\S+",
    r"password\s*[:=]\s*\S+",
    r"secret\s*[:=]\s*\S+",
]

def has_temporal_correlation_only(evidence):
    temporal_phrases = [
        "started after",
        "occurred after",
        "began after",
        "following release",
        "after deployment"
    ]

    text = evidence.lower()

    return any(
        phrase in text
        for phrase in temporal_phrases
    )


def contains_sensitive_data(text):
    for pattern in SECRET_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False