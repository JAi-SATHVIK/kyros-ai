import re
import os

# Common greetings, acknowledgments, and conversational fillers
DEFAULT_NON_FACTUAL_PATTERNS = [
    r"^(hello|hi|hey|howdy|sup|greetings|good\s+(morning|afternoon|evening))\b",
    r"^(ok|okay|yes|yep|yup|sure|cool|fine|correct|perfect|no|nope|nah|thanks|thank\s+you|great|awesome|understood)\b",
    r"^(how\s+are\s+you|what's\s+up|how's\s+it\s+going)\??$",
    r"^(testing|test|ping|pong)\b"
]

# Structural indicators that a text contains factual weight (e.g. named details, possessives, quantities)
FACTUAL_INDICATORS = [
    r"\b(my|his|her|their|our|its)\s+\w+",             # Possessives indicating relationships/properties
    r"\b(prefer|like|dislike|love|hate|want|need)\b",   # Preferences
    r"\b(live|work|born|study|employee|manager|ceo)\b", # Status/profile attributes
    r"\b(at|in|on)\s+[A-Z]\w+",                         # Locations/Organizations with proper capitalizations
    r"\b\d{1,4}\b",                                     # Numbers/years/dates
    r"\b(tomorrow|yesterday|next\s+week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b" # Temporal context
]

_compiled_non_factual = [re.compile(pattern, re.IGNORECASE) for pattern in DEFAULT_NON_FACTUAL_PATTERNS]
_compiled_factual_indicators = [re.compile(pattern, re.IGNORECASE) for pattern in FACTUAL_INDICATORS]

def get_custom_patterns(env_var: str) -> list[re.Pattern]:
    """Parse custom regex lists from environmental variables."""
    raw = os.environ.get(env_var, "")
    if not raw:
        return []
    patterns = []
    for p in raw.split(","):
        p_clean = p.strip()
        if p_clean:
            try:
                patterns.append(re.compile(p_clean, re.IGNORECASE))
            except re.error:
                pass
    return patterns

def is_factual_content(text: str) -> bool:
    """Evaluate whether the content contains factual or entity-rich information.
    
    Uses heuristics, custom environmental whitelists/blacklists, and structural indicators.
    """
    if not text:
        return False
        
    cleaned = text.strip()
    words = cleaned.split()
    
    # 1. Custom Blacklist / Whitelist Check
    custom_whitelist = get_custom_patterns("KYROS_FACTUAL_WHITELIST")
    for pattern in custom_whitelist:
        if pattern.search(cleaned):
            return True
            
    custom_blacklist = get_custom_patterns("KYROS_FACTUAL_BLACKLIST")
    for pattern in custom_blacklist:
        if pattern.search(cleaned):
            return False

    # 2. Check for strong structural factual indicators
    for indicator in _compiled_factual_indicators:
        if indicator.search(cleaned):
            return True

    # 3. Very short statements (<= 2 words) are rarely factual or rich enough
    if len(words) <= 2:
        for pattern in _compiled_non_factual:
            if pattern.search(cleaned):
                return False
        # If it's a short statement but not a greeting/filler, keep it if it has minimum character length
        return len(cleaned) > 2
        
    # 4. Check regular pattern matches for conversational filler
    for pattern in _compiled_non_factual:
        if pattern.match(cleaned):
            # If a sentence starts with a greeting but is long/has more context, it might be factual
            if len(words) > 6:
                continue
            return False
            
    return True
