"""V10+V11+V12 — Memory classification, importance scoring, and semantic extraction.

V10: Lightweight classifier — determines if content is episodic/semantic/procedural
V11: Auto-store with importance scoring
V12: Semantic triple extractor — detects subject-predicate-object facts
"""

from __future__ import annotations

import re
from enum import Enum
from dataclasses import dataclass

from kyros.logging import get_logger

_logger = get_logger("kyros.proxy.classifier")


class MemoryCategory(str, Enum):
    """Classification of extracted memory content."""
    EPISODIC = "episodic"       # Events, conversations, observations
    SEMANTIC = "semantic"       # Facts, preferences, truths
    PROCEDURAL = "procedural"   # How-to, workflows, steps
    SKIP = "skip"               # Not worth storing


@dataclass
class ClassificationResult:
    """Result of the memory classifier."""
    category: MemoryCategory
    importance: float          # 0.0 – 1.0
    confidence: float          # How confident we are in the classification
    reason: str                # Human-readable reason


@dataclass
class SemanticTriple:
    """A subject-predicate-object fact extracted from text."""
    subject: str
    predicate: str
    obj: str                   # 'object' is a Python builtin
    confidence: float


# ─── V10: Lightweight Classifier ──────────────

# Keyword patterns for each category (fast, no LLM needed)
_SEMANTIC_SIGNALS = [
    r"\b(?:is|are|was|were)\s+(?:a|an|the)\b",        # "X is a Y"
    r"\b(?:works?\s+(?:at|for|in))\b",                # "works at company"
    r"\b(?:lives?\s+in)\b",                            # "lives in city"
    r"\b(?:prefers?|likes?|wants?|loves?|hates?)\b",   # Preferences
    r"\b(?:born|founded|created|started)\b",           # Facts
    r"\b(?:email|phone|address|name)\s+is\b",          # Identity facts
    r"\b(?:always|never|every|usually)\b",             # Habitual patterns
    r"\b(?:costs?|priced?\s+at|worth)\b",              # Pricing facts
]

_PROCEDURAL_SIGNALS = [
    r"\b(?:step\s+\d|first|then|next|finally)\b",     # Sequential steps
    r"\b(?:to\s+do\s+this|how\s+to|in\s+order\s+to)\b",
    r"\b(?:run|execute|install|deploy|build|configure)\b",
    r"\b(?:workflow|process|procedure|pipeline)\b",
    r"\b(?:click|type|enter|select|choose)\b",         # UI instructions
]

_SKIP_SIGNALS = [
    r"^(?:ok|okay|sure|yes|no|thanks|thank you|got it|alright)[.!]?$",
    r"^(?:hello|hi|hey|good\s+(?:morning|afternoon|evening))",
    r"^(?:how\s+can\s+i\s+help|is\s+there\s+anything\s+else)",
    r"^(?:you're\s+welcome|no\s+problem|my\s+pleasure)",
    r"^(?:i\s+don'?t\s+know|i'?m\s+not\s+sure|i\s+cannot)",
]

_IMPORTANCE_BOOSTERS = [
    (r"\b(?:important|critical|urgent|deadline|must|required)\b", 0.15),
    (r"\b(?:password|secret|key|token|credential)\b", 0.20),
    (r"\b(?:error|bug|issue|problem|fail)\b", 0.10),
    (r"\b(?:decided|agreed|confirmed|approved)\b", 0.15),
    (r"\b(?:name|email|phone|address|company)\b", 0.10),
    (r"\$\d+", 0.10),            # Dollar amounts
    (r"\b\d{4}-\d{2}-\d{2}\b", 0.05),  # Dates
]


def classify_content(content: str) -> ClassificationResult:
    """Classify text content into memory categories.

    Uses keyword pattern matching for speed (<0.1ms per call).
    No LLM inference required — runs entirely on regex.

    Args:
        content: The text to classify.

    Returns:
        ClassificationResult with category, importance, and confidence.
    """
    if not content or len(content.strip()) < 10:
        return ClassificationResult(
            category=MemoryCategory.SKIP,
            importance=0.0,
            confidence=1.0,
            reason="Content too short",
        )

    text_lower = content.strip().lower()

    # Check skip patterns first
    for pattern in _SKIP_SIGNALS:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return ClassificationResult(
                category=MemoryCategory.SKIP,
                importance=0.0,
                confidence=0.95,
                reason="Generic/trivial response",
            )

    # Score each category
    semantic_score = sum(
        1 for p in _SEMANTIC_SIGNALS if re.search(p, text_lower, re.IGNORECASE)
    )
    procedural_score = sum(
        1 for p in _PROCEDURAL_SIGNALS if re.search(p, text_lower, re.IGNORECASE)
    )

    # Calculate importance
    base_importance = 0.5
    for pattern, boost in _IMPORTANCE_BOOSTERS:
        if re.search(pattern, content, re.IGNORECASE):
            base_importance = min(1.0, base_importance + boost)

    # Length-based importance adjustment (longer = usually more important)
    word_count = len(content.split())
    if word_count > 50:
        base_importance = min(1.0, base_importance + 0.10)
    elif word_count < 10:
        base_importance = max(0.1, base_importance - 0.15)

    # Determine category
    if procedural_score >= 2 and procedural_score > semantic_score:
        return ClassificationResult(
            category=MemoryCategory.PROCEDURAL,
            importance=min(1.0, base_importance + 0.1),
            confidence=min(0.95, 0.5 + procedural_score * 0.1),
            reason=f"Procedural signals detected ({procedural_score} matches)",
        )

    if semantic_score >= 2:
        return ClassificationResult(
            category=MemoryCategory.SEMANTIC,
            importance=min(1.0, base_importance + 0.05),
            confidence=min(0.95, 0.5 + semantic_score * 0.1),
            reason=f"Semantic/factual signals detected ({semantic_score} matches)",
        )

    # Default: episodic (conversations, events)
    return ClassificationResult(
        category=MemoryCategory.EPISODIC,
        importance=base_importance,
        confidence=0.6,
        reason="Default classification: episodic/conversational content",
    )


# ─── V12: Semantic Triple Extractor ───────────

# Patterns for extracting subject-predicate-object triples
_TRIPLE_PATTERNS = [
    # "X is a Y" / "X is Y"
    (r"(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:is|are)\s+(?:a|an|the)\s+(.+?)(?:\.|,|$)",
     "is_a"),
    # "X works at Y"
    (r"(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+works?\s+(?:at|for)\s+(.+?)(?:\.|,|$)",
     "works_at"),
    # "X lives in Y"
    (r"(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+lives?\s+in\s+(.+?)(?:\.|,|$)",
     "lives_in"),
    # "X prefers Y" / "X likes Y"
    (r"(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:prefers?|likes?|wants?|loves?)\s+(.+?)(?:\.|,|$)",
     "prefers"),
    # "X's email is Y"
    (r"(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)'?s?\s+(email|phone|name|address|role|title)\s+is\s+(.+?)(?:\.|,|$)",
     None),  # Special: subject from group 1, predicate from group 2, object from group 3
    # "The user prefers Y"
    (r"[Tt]he\s+(?:user|customer|client)\s+(?:prefers?|likes?|wants?)\s+(.+?)(?:\.|,|$)",
     "prefers"),
    # "User preference: Y"
    (r"[Uu]ser\s+(?:preference|setting|choice):\s*(.+?)(?:\.|,|$)",
     "preference"),
]


def extract_triples(content: str) -> list[SemanticTriple]:
    """Extract subject-predicate-object triples from text.

    Uses regex patterns to find factual statements. Fast (<0.5ms)
    and does not require LLM inference.

    Args:
        content: Text to extract facts from.

    Returns:
        List of SemanticTriple objects found in the text.
    """
    triples = []

    for pattern, default_predicate in _TRIPLE_PATTERNS:
        for match in re.finditer(pattern, content):
            groups = match.groups()

            if default_predicate is None and len(groups) == 3:
                # Special pattern: subject, predicate, object in groups
                triple = SemanticTriple(
                    subject=groups[0].strip(),
                    predicate=groups[1].strip().lower(),
                    obj=groups[2].strip(),
                    confidence=0.75,
                )
            elif len(groups) == 2 and default_predicate:
                triple = SemanticTriple(
                    subject=groups[0].strip(),
                    predicate=default_predicate,
                    obj=groups[1].strip(),
                    confidence=0.70,
                )
            elif len(groups) == 1 and default_predicate:
                triple = SemanticTriple(
                    subject="user",
                    predicate=default_predicate,
                    obj=groups[0].strip(),
                    confidence=0.65,
                )
            else:
                continue

            # Skip if object is too long (likely a sentence, not a fact)
            if len(triple.obj.split()) > 8:
                continue

            triples.append(triple)

    return triples


# ─── V11: Auto-Store with Importance Scoring ──

async def auto_store_classified(
    agent_id: str,
    content: str,
    role: str,
    config,  # ProxyConfig
) -> dict | None:
    """Classify content, score importance, and store appropriately.

    Combines V10 (classification) + V11 (importance scoring) + V12 (triple extraction)
    into a single pipeline that runs after every LLM response.

    Args:
        agent_id: The agent to store memories for.
        content: The text content to process.
        role: "user" or "assistant".
        config: ProxyConfig instance.

    Returns:
        Dict with storage results, or None if content was skipped.
    """
    import httpx

    # V10: Classify
    classification = classify_content(content)

    if classification.category == MemoryCategory.SKIP:
        return None

    # V12: Extract semantic triples (for semantic facts)
    triples = []
    if classification.category == MemoryCategory.SEMANTIC:
        triples = extract_triples(content)

    result = {"classification": classification.category.value, "stored": []}

    # Store episodic memory (always, if not skipped)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{config.kyros_base_url}/v1/memory/episodic/remember",
                json={
                    "agent_id": agent_id,
                    "content": content,
                    "importance": classification.importance,
                },
                headers={
                    "X-API-Key": config.kyros_api_key,
                    "Content-Type": "application/json",
                },
            )
            if resp.status_code in (200, 201):
                result["stored"].append({
                    "type": "episodic",
                    "memory_id": resp.json().get("memory_id"),
                })
            else:
                _logger.warning("Auto-store episodic failed", status=resp.status_code, agent_id=agent_id)
    except Exception as e:
        _logger.error("Auto-store episodic error", error=str(e), agent_id=agent_id)

    # V12: Store extracted semantic triples as facts
    for triple in triples:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{config.kyros_base_url}/v1/memory/semantic/facts",
                    json={
                        "agent_id": agent_id,
                        "subject": triple.subject,
                        "predicate": triple.predicate,
                        "object": triple.obj,
                        "confidence": triple.confidence,
                    },
                    headers={
                        "X-API-Key": config.kyros_api_key,
                        "Content-Type": "application/json",
                    },
                )
                if resp.status_code in (200, 201):
                    result["stored"].append({
                        "type": "semantic",
                        "triple": f"{triple.subject} → {triple.predicate} → {triple.obj}",
                    })
                else:
                    _logger.warning("Auto-store triple failed", status=resp.status_code, agent_id=agent_id)
        except Exception as e:
            _logger.error("Auto-store triple error", error=str(e), agent_id=agent_id)

    return result
