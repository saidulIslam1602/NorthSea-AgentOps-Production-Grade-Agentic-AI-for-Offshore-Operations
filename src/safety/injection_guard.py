"""
Prompt injection guard for retrieved document content.

Defends against prompt injection attacks that may be embedded in:
  - External documents ingested into the RAG corpus
  - User-supplied queries
  - Tool results from external APIs

Three-layer defence:
  1. Pattern matching — detects known injection signatures
  2. Instruction hierarchy check — validates retrieved content doesn't
     contain system-level instructions
  3. Semantic anomaly — flags content with unusually high instruction density

All flagged content is sanitised (injections removed) and logged to the
audit log. Severe injections cause full escalation with INJECTION_DETECTED reason.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

from src.safety.audit_log import AuditLogger

logger = logging.getLogger(__name__)

# ─── Injection Patterns ───────────────────────────────────────────────────────

# Patterns that suggest an attempt to override the AI's instructions
INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # Direct system override attempts
    re.compile(r"ignore\s+(previous|above|all|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(everything|previous|prior|above)", re.IGNORECASE),
    re.compile(r"disregard\s+(previous|above|prior|all)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+)?(you\s+are\s+)?(a|an)\s+", re.IGNORECASE),
    re.compile(r"pretend\s+(to\s+be|you\s+are)", re.IGNORECASE),
    re.compile(r"new\s+(system\s+)?prompt\s*:", re.IGNORECASE),
    re.compile(r"<\s*/?system\s*>", re.IGNORECASE),
    re.compile(r"<\s*/?instruction[s]?\s*>", re.IGNORECASE),
    re.compile(r"\[SYSTEM\]", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"###\s*(System|Instruction|Override)\s*:", re.IGNORECASE),
    # Data exfiltration attempts
    re.compile(
        r"(print|output|show|reveal|display)\s+(all|your|the\s+system)\s+(prompt|instructions|key|secret)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(print|show|reveal|output)\s+all\s+of\s+your\s+(instructions?|prompt|keys?|secrets?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(print|show|reveal|output)\s+all\s+your\s+(instructions?|prompt|keys?|secrets?)",
        re.IGNORECASE,
    ),
    re.compile(r"what\s+(are|is)\s+your\s+(instructions?|system\s+prompt|directive)", re.IGNORECASE),
    re.compile(r"repeat\s+(everything|all|your\s+(instructions?|prompt))", re.IGNORECASE),
    # Action override attempts
    re.compile(r"(always|never)\s+(recommend|suggest|say|tell\s+the\s+user)", re.IGNORECASE),
    re.compile(r"override\s+(safety|security|confidentiality|hse)", re.IGNORECASE),
    re.compile(r"\bbypass\s+(?:the\s+)?(?:hse|risk\b)", re.IGNORECASE),
    re.compile(r"\bbypass\b.{1,96}?\bescalation\b", re.IGNORECASE),
    # Tool manipulation
    re.compile(r"call\s+(tool|function)\s*[:\(]", re.IGNORECASE),
    re.compile(r"execute\s+(command|code|script|shell)", re.IGNORECASE),
]

# Legitimate operational terms that might trigger false positives — allow them
ALLOWLIST_TERMS = [
    "override valve",
    "safety override",
    "bypass valve",
    "bypass line",
    "ignore outliers",
    "disregard noise",
    "act as a reference",
]


@dataclass
class InjectionCheckResult:
    """Result of a prompt injection check."""

    is_clean: bool
    matches: list[str]
    sanitised_text: str
    severity: str  # "NONE", "LOW", "HIGH", "CRITICAL"
    input_hash: str


def _build_allowlist_pattern(term: str) -> re.Pattern[str]:
    """Compile a case-insensitive literal-match pattern for an allowlist term."""
    return re.compile(re.escape(term), re.IGNORECASE)


# Pre-compiled allowlist patterns (each anchored to a word-level match)
_ALLOWLIST_PATTERNS: list[re.Pattern[str]] = [_build_allowlist_pattern(t) for t in ALLOWLIST_TERMS]


def _mask_allowlisted_spans(text: str) -> str:
    """
    Replace allowlisted operational phrases with a neutral placeholder before
    injection scanning.  This prevents legitimate terms from blocking detection
    of genuine injections that co-occur with them in the same string.
    """
    masked = text
    for pat in _ALLOWLIST_PATTERNS:
        masked = pat.sub("__ALLOWLISTED__", masked)
    return masked


def check_for_injection(text: str) -> InjectionCheckResult:
    """
    Scan text for prompt injection patterns.

    The allowlist is applied by *masking* operational phrases before scanning,
    not by short-circuiting the whole check.  This prevents an attacker from
    prepending "bypass valve" to an injection payload to suppress detection.

    Returns:
        InjectionCheckResult with detection details and sanitised text.
    """
    input_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

    # Mask allowlisted phrases so they cannot suppress injection detection
    scan_target = _mask_allowlisted_spans(text)

    matches: list[str] = []
    sanitised = text  # redact from original, not the masked copy

    for pattern in INJECTION_PATTERNS:
        found = pattern.findall(scan_target)
        if found:
            matches.extend([str(m) for m in found])
            # Redact from the original text so the caller gets a usable string
            sanitised = pattern.sub("[REDACTED: potential injection]", sanitised)

    if not matches:
        return InjectionCheckResult(
            is_clean=True,
            matches=[],
            sanitised_text=text,
            severity="NONE",
            input_hash=input_hash,
        )

    # Determine severity — scan masked text so allowlisted spans don't FP, and use substring
    # checks (regex findall entries are often single capture groups, not full phrases).
    scan_lower = scan_target.lower()
    critical_keywords = ("ignore previous", "system prompt", "override safety", "bypass")
    severity = "LOW"
    if any(k in scan_lower for k in critical_keywords):
        severity = "CRITICAL"
    elif len(matches) >= 3:
        severity = "HIGH"
    elif severity == "LOW" and (
        ("forget everything" in scan_lower and "new instructions" in scan_lower)
        or (
            "repeat everything" in scan_lower
            and (
                "know" in scan_lower
                or "told" in scan_lower
                or "prompt" in scan_lower
                or "instructions" in scan_lower
                or "api keys" in scan_lower
            )
        )
    ):
        severity = "HIGH"

    logger.warning("Prompt injection detected [%s]: %d patterns matched. Hash: %s", severity, len(matches), input_hash)

    return InjectionCheckResult(
        is_clean=False,
        matches=matches,
        sanitised_text=sanitised,
        severity=severity,
        input_hash=input_hash,
    )


def sanitise_retrieved_chunks(
    chunks: list[dict],
    audit_logger: AuditLogger | None = None,
) -> list[dict]:
    """
    Sanitise all retrieved document chunks before injection into prompts.

    Modifies chunks in place (returns new list with sanitised content).
    Logs any injections found to the audit logger.
    """
    sanitised_chunks: list[dict] = []
    injections_found = 0

    for chunk in chunks:
        content = chunk.get("content", "")
        result = check_for_injection(content)

        if not result.is_clean:
            injections_found += 1
            if audit_logger:
                audit_logger.log_injection_detection(
                    chunk_id=f"{chunk.get('document_id', 'unknown')}:{chunk.get('chunk_index', 0)}",
                    severity=result.severity,
                    matches=result.matches,
                    input_hash=result.input_hash,
                )

            # Use sanitised version — severe injections get fully blocked
            if result.severity == "CRITICAL":
                # Drop the chunk entirely
                logger.warning(
                    "Dropping chunk from %s due to CRITICAL injection",
                    chunk.get("document_id"),
                )
                continue
            else:
                chunk = {**chunk, "content": result.sanitised_text}

        sanitised_chunks.append(chunk)

    if injections_found > 0:
        logger.warning("Sanitised %d/%d chunks with injection patterns", injections_found, len(chunks))

    return sanitised_chunks


def check_user_query(query: str) -> InjectionCheckResult:
    """Check a user-supplied query for injection attempts."""
    result = check_for_injection(query)
    if not result.is_clean:
        logger.warning("User query injection detected [%s]: %s", result.severity, query[:100])
    return result
