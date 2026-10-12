import re


SECRET_PATTERNS = [
    r"\bsk-[A-Za-z0-9_-]{8,}\b",                         # API-key-like value
    r"\bapi[_-]?key\s*[:=]\s*[^\s,;]+",                 # API key assignments
    r"\b(password|passwd|secret|access[_-]?token)\s*[:=]\s*[^\s,;]+",
    r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}",              # bearer token
    r"\bAKIA[0-9A-Z]{16}\b",                            # AWS access key ID
    r"\bgh[pousr]_[A-Za-z0-9]{20,}\b",                   # GitHub token family
]

PII_PATTERNS = {
    "email_address": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    "phone_number": r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)",
    "aadhaar_like_identifier": r"(?<!\d)(?:\d{4}[\s-]?){2}\d{4}(?!\d)",
    "pan_like_identifier": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "ssn_like_identifier": r"\b\d{3}-\d{2}-\d{4}\b",
}


def detect_sensitive_data(text):
    """Return detected data categories only; never return matched secret values."""
    value = str(text or "")
    categories = []
    if any(re.search(pattern, value, re.IGNORECASE) for pattern in SECRET_PATTERNS):
        categories.append("secret_or_credential")
    for category, pattern in PII_PATTERNS.items():
        if re.search(pattern, value, re.IGNORECASE):
            categories.append(category)
    return categories


def contains_sensitive_data(text):
    """Backward-compatible boolean detector for credentials and common structured PII."""
    return bool(detect_sensitive_data(text))

CAUSAL_MARKERS = [
    "caused",
    "cause",
    "root cause",
    "due to",
    "because of",
    "result of",
    "triggered by",
    "responsible for",
]

NEGATION_PATTERNS = [
    "no evidence that",
    "there is no evidence that",
    "no clear evidence",
    "not established",
    "not proven",
    "does not establish",
    "does not prove",
    "cannot conclude",
    "insufficient evidence",
    "not enough evidence",
    "no indication that",
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




def guardrail_llm_output(text, incident_id="INC001", release_id="Release 4.3"):
    """Reject unsupported causal claims while allowing clearly negated statements."""
    if text is None:
        return text

    normalized = str(text)
    lowered = normalized.lower()

    if not any(marker in lowered for marker in CAUSAL_MARKERS):
        return normalized

    if any(phrase in lowered for phrase in NEGATION_PATTERNS):
        return normalized

    return (
        f"The evidence shows that {incident_id} started after {release_id}, "
        "but causation has not been established. Further investigation is required."
    )

INSTRUCTION_PATTERNS = [
    r"\bignore\s+(previous|prior|all)\s+instructions\b",
    r"\bignore\s+the\s+instructions\b",
    r"\breveal\b.*\b(api key|password|secret|token)\b",
    r"\bstate\s+that\b",
    r"\bclaim\s+that\b",
    r"\bsay\s+that\b",
    r"\btell\s+the\s+user\b",
    r"\bdo\s+not\s+follow\b",
    r"\bdisregard\b.*\binstructions\b",
]

def contains_instruction(text):
    """Detect known instruction-like phrases; a non-match does not certify safety."""
    text = (text or "").lower()

    for pattern in INSTRUCTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False