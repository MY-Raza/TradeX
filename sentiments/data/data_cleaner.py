"""
reddit_crypto_cleaner.py
========================
Production-ready text cleaning module for Reddit crypto sentiment analysis.
Optimized for: cardiffnlp/twitter-roberta-base-sentiment

Design philosophy:
  - Surgical removal of structural noise only
  - Zero tolerance for over-cleaning (emojis, caps, slang, punctuation are SIGNALS)
  - Batch-safe, pandas-compatible, zero heavy NLP dependencies
  - Integrated feature extraction for downstream ML
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import pandas as pd


# ================================================================================
# COMPILED REGEX PATTERNS
# (Compiled once at import time — safe for batch use across millions of rows)
# ================================================================================

# --- Hard noise ---
_RE_URL = re.compile(
    r"http[s]?://\S+|www\.\S+|\[.*?\]\(https?://\S+\)",
    re.IGNORECASE
)
_RE_HTML_ENTITY = re.compile(r"&(?:#\d+|[a-zA-Z]+);")
_RE_HTML_TAG = re.compile(r"<[^>]+>")

# Markdown artifacts that become garbage tokens for the tokenizer
_RE_MARKDOWN_BOLD_ITALIC = re.compile(r"\*{1,3}(.*?)\*{1,3}")   # **bold** / *italic*
_RE_MARKDOWN_STRIKE = re.compile(r"~~(.*?)~~")                   # ~~strikethrough~~
_RE_MARKDOWN_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)    # ```code blocks```
_RE_MARKDOWN_INLINE_CODE = re.compile(r"`[^`]+`")                # `inline code`
_RE_MARKDOWN_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)   # ## Headers
_RE_MARKDOWN_QUOTE = re.compile(r"^>+\s?", re.MULTILINE)        # > blockquotes
_RE_MARKDOWN_HR = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)   # --- horizontal rules
_RE_MARKDOWN_TABLE_SEP = re.compile(r"^\|?[-| :]+\|$", re.MULTILINE)  # table separators

# --- Reddit-specific structural noise ---
# Matches "[deleted]", "[removed]", "[Dead]" etc. as standalone content
_RE_REDDIT_REMOVED = re.compile(
    r"^\s*\[(?:deleted|removed|dead|unavailable)\]\s*$",
    re.IGNORECASE
)
# Common bot signatures
_RE_BOT_PATTERNS = re.compile(
    r"(?:"
    r"i am a bot[,\s].*?(?:contact the moderators|modmail)|"
    r"this action was performed automatically|"
    r"please contact the moderators|"
    r"if you have any questions.*?message the mods|"
    r"^bot\s*\||"                               # "Bot | subreddit name" headers
    r"automoderator|"
    r"i'm a bot\b"
    r")",
    re.IGNORECASE | re.DOTALL
)
# Subreddit mod boilerplate (common across crypto subs)
_RE_MOD_BOILERPLATE = re.compile(
    r"(?:"
    r"please read our.*?rules|"
    r"this is not financial advice|"
    r"^reminder[:\s].*?rules|"
    r"not your keys[,\s]+not your coins?|"  # common copypasta — not a signal
    r"do your own research[\s.!]*dyor"       # pure boilerplate, no sentiment
    r")",
    re.IGNORECASE
)

# --- Spam / low-quality patterns ---
# Repeated chars: "MOOOOON" → "MOON", "lolololol" → "lolol" (keep 4, collapse rest)
# We collapse runs > 4 of the same character down to 4.
# NOTE: We keep up to 4 so "!!!!" remains but "!!!!!!!!" → "!!!!"
_RE_CHAR_REPEAT = re.compile(r"(.)\1{4,}")

# Ultra-short zero-signal phrases (only matched if they are the ENTIRE text)
_RE_ZERO_SIGNAL = re.compile(
    r"^(?:ok|okay|lol|lmao|lmfao|haha|hehe|yep|yup|nope|nah|yes|no|wow|"
    r"same|nice|true|k|ty|thx|thanks|np|gg|rip|f|oof|meh|hmm|smh|ikr|"
    r"idk|imo|tbh|irl|omg|omfg|wtf|ffs|fml|bruh|bro|dude|ditto|based|cringe)$",
    re.IGNORECASE
)

# Crypto ticker normalization: $BTC → BTC (removes $ noise token, keeps reference)
_RE_TICKER = re.compile(r"\$([A-Z]{2,10})\b")

# Price normalization patterns (optional, off by default)
_RE_PRICE_K = re.compile(r"\$?([\d,]+\.?\d*)[kK]\b")           # 50k → 50000
_RE_PRICE_M = re.compile(r"\$?([\d,]+\.?\d*)[mM]\b")           # 2.5M → 2500000

# Whitespace normalization (collapse all whitespace runs including \n \t to single space)
_RE_WHITESPACE = re.compile(r"\s+")

# For feature extraction
_RE_EMOJI = re.compile(
    "[\U00002600-\U000027BF"      # Misc symbols
    "\U0001F300-\U0001F9FF"       # All common emoji blocks
    "\U00002700-\U000027BF"       # Dingbats
    "\U0001FA00-\U0001FA9F"       # Chess symbols / new emoji
    "\U00002500-\U00002BEF]+",    # Box drawing etc.
    flags=re.UNICODE
)
_RE_CAPS_WORD = re.compile(r"\b[A-Z]{2,}\b")    # ALL-CAPS words (min 2 chars)
_RE_PUNCT_INTENSITY = re.compile(r"[!?]{2,}")   # !! ??? etc.


# ================================================================================
# CRYPTO SLANG PRESERVATION REGISTRY
# (Documented here for auditability — NOT used for removal, only for reference
#  and the spam-score feature. The cleaner does NOT touch these tokens.)
# ================================================================================
CRYPTO_SLANG_TOKENS: frozenset[str] = frozenset({
    "hodl", "hodler", "hodling",
    "fomo", "fud", "rekt", "ngmi", "wagmi", "gm", "gn",
    "wen", "ser", "fren", "anon",
    "moon", "mooning", "moonshot",
    "dump", "dumping", "pump", "pumping", "pamp",
    "dip", "dyor", "nfa",
    "diamond", "hands", "paper", "weak",
    "sats", "satoshi", "plebs", "cope", "cope harder",
    "ape", "aping", "degen", "degens",
    "lambo", "laser", "eyes",
    "bear", "bull", "bagholder", "bag",
    "alt", "alts", "altcoin", "altseason",
    "maxi", "maximalist", "shitcoin", "shitcoins",
})


# ================================================================================
# FEATURE EXTRACTION DATACLASS
# ================================================================================
@dataclass
class TextFeatures:
    """
    Auxiliary features extracted alongside cleaned text.
    Use these as additional signals in your trading ML model
    — they quantify emotional intensity that the sentiment
    score alone may not capture.
    """
    cleaned_text: str
    is_valid: bool                       # False → row should be dropped before inference

    # Intensity signals
    emoji_count: int = 0                 # Raw emoji character count
    caps_ratio: float = 0.0             # Fraction of alpha chars that are uppercase
    punct_intensity: int = 0            # Count of !! / ??? clusters
    exclamation_count: int = 0          # Raw ! count
    question_count: int = 0             # Raw ? count

    # Quality signals
    char_repeat_hits: int = 0           # Times the char-repeat rule fired
    spam_score: float = 0.0             # [0, 1] composite spam proxy
    token_count: int = 0                # Whitespace-tokenized word count

    # Deduplication
    content_hash: str = ""              # MD5 of cleaned_text (exact dedup)
    simhash: int = 0                    # 64-bit SimHash (near-dedup)

    # Flags
    had_url: bool = False
    had_bot_content: bool = False
    had_markdown: bool = False


# ================================================================================
# SIMHASH IMPLEMENTATION
# (Pure Python — no dependencies. O(n_tokens) per text.)
# ================================================================================

def _simhash(text: str, n_bits: int = 64) -> int:
    """
    Compute a 64-bit SimHash fingerprint for near-duplicate detection.

    Two texts are near-duplicates if their SimHash Hamming distance ≤ 3.
    This catches copy-paste variants, minor edits, and quoted reposts
    without requiring embedding similarity (which is prohibitively slow at scale).

    Args:
        text: Input text (should be the cleaned version)
        n_bits: Hash width (64 is standard)

    Returns:
        int: 64-bit SimHash integer
    """
    tokens = text.lower().split()
    if not tokens:
        return 0

    v = [0] * n_bits

    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for i in range(n_bits):
            bit = (h >> i) & 1
            v[i] += 1 if bit else -1

    fingerprint = 0
    for i in range(n_bits):
        if v[i] > 0:
            fingerprint |= 1 << i

    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Compute bitwise Hamming distance between two SimHash integers."""
    return bin(a ^ b).count("1")


# ================================================================================
# CORE CLEANING HELPERS
# ================================================================================

def _strip_markdown(text: str) -> tuple[str, bool]:
    """
    Strip Reddit markdown artifacts while preserving the underlying text content.

    We extract the TEXT from markdown constructs rather than deleting them entirely
    because "**BUY NOW**" should become "BUY NOW" (the sentiment signal survives),
    not "" (signal destroyed).

    Returns:
        (cleaned_text, had_markdown: bool)
    """
    had = False

    # Code blocks first (content inside is code, not sentiment — remove entirely)
    if _RE_MARKDOWN_CODE_BLOCK.search(text):
        text = _RE_MARKDOWN_CODE_BLOCK.sub(" ", text)
        had = True

    # Inline code: remove entirely
    if _RE_MARKDOWN_INLINE_CODE.search(text):
        text = _RE_MARKDOWN_INLINE_CODE.sub(" ", text)
        had = True

    # Bold/italic: keep inner text
    new = _RE_MARKDOWN_BOLD_ITALIC.sub(r"\1", text)
    if new != text:
        text, had = new, True

    # Strikethrough: keep inner text (it still carries sentiment context)
    new = _RE_MARKDOWN_STRIKE.sub(r"\1", text)
    if new != text:
        text, had = new, True

    # Headers: strip the # prefix, keep the header text
    new = _RE_MARKDOWN_HEADER.sub("", text)
    if new != text:
        text, had = new, True

    # Block quotes: strip the > prefix, keep quoted text
    new = _RE_MARKDOWN_QUOTE.sub("", text)
    if new != text:
        text, had = new, True

    # Horizontal rules / table separators: remove entirely
    new = _RE_MARKDOWN_HR.sub(" ", text)
    new = _RE_MARKDOWN_TABLE_SEP.sub(" ", new)
    if new != text:
        text, had = new, True

    return text, had


def _is_bot_or_removed(text: str) -> bool:
    """Return True if the text is a deleted post or bot message."""
    if _RE_REDDIT_REMOVED.match(text):
        return True
    if _RE_BOT_PATTERNS.search(text):
        return True
    return False


def _is_zero_signal(text: str) -> bool:
    """
    Return True only if the entire text is a zero-signal filler phrase
    AND contains no emojis, caps words, or punctuation intensity.

    We deliberately require ALL conditions to be false before discarding:
    "lol 🚀" → NOT zero signal (has emoji)
    "LOL"    → NOT zero signal (has caps)
    "lol!!!" → NOT zero signal (has punct intensity)
    "lol"    → zero signal → discard
    """
    if not _RE_ZERO_SIGNAL.match(text.strip()):
        return False
    if _RE_EMOJI.search(text):
        return False
    if _RE_CAPS_WORD.search(text):
        return False
    if _RE_PUNCT_INTENSITY.search(text):
        return False
    return True


# ================================================================================
# FEATURE EXTRACTORS
# ================================================================================

def _count_emojis(text: str) -> int:
    return sum(len(m.group()) for m in _RE_EMOJI.finditer(text))


def _caps_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are uppercase."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def _spam_score(text: str, char_repeat_hits: int, token_count: int) -> float:
    """
    Composite spam score in [0, 1].

    Heuristics (all independent, max-capped):
      - Very short texts that passed zero-signal filter: mild penalty
      - High character-repeat hits relative to token count
      - Extremely high caps ratio (all-caps long text can be a sign of spam)
      - High ratio of punctuation to total chars

    This is a PROXY, not a classifier. Use as an auxiliary feature, not a hard filter.
    """
    score = 0.0

    if token_count < 3:
        score += 0.2

    if token_count > 0:
        repeat_ratio = char_repeat_hits / token_count
        score += min(repeat_ratio * 0.3, 0.3)

    cr = _caps_ratio(text)
    if cr > 0.8 and token_count > 5:
        score += 0.2

    if len(text) > 0:
        punct_chars = sum(1 for c in text if c in "!?.,;:")
        punct_ratio = punct_chars / len(text)
        if punct_ratio > 0.4:
            score += 0.3

    return min(score, 1.0)


# ================================================================================
# MAIN CLEANING FUNCTION
# ================================================================================

def clean_text(
    text,
    normalize_tickers: bool = True,
    normalize_prices: bool = False,
    max_length: int = 512,
) -> str:
    """
    Production-ready text cleaner for Reddit crypto sentiment inference.

    Optimized for: cardiffnlp/twitter-roberta-base-sentiment

    This function is the drop-in replacement for the existing clean_text() in
    sentiment_analysis.py. It is stateless and side-effect-free — safe for
    parallel / vectorized application via df[col].apply(clean_text).

    Args:
        text:               Raw text from Reddit post title or comment body.
        normalize_tickers:  Replace $BTC → BTC (removes noise $ token). Default True.
        normalize_prices:   Replace 50k → 50000. Default False (transforms too many tokens).
        max_length:         Hard truncation in characters before tokenization.
                            RoBERTa max is 512 tokens ≈ 1500–2000 chars; 512 chars
                            is conservative and safe. Adjust if titles are being cut.

    Returns:
        str: Cleaned text ready for tokenization, or "" if the text should be
             dropped (deleted post, bot message, zero-signal content).
    """
    # ── 0. Null / type guard ──────────────────────────────────────────────────
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).strip()
    if not text:
        return ""

    # ── 1. Hard gate: deleted / removed / bot messages ───────────────────────
    # These contain zero sentiment signal and must be excluded BEFORE any
    # other processing to avoid false neutrals polluting the dataset.
    if _is_bot_or_removed(text):
        return ""

    # ── 2. HTML entities and tags ─────────────────────────────────────────────
    # Reddit's API sometimes returns HTML-escaped text (&amp; &gt; etc.)
    # and occasional HTML tags in older post bodies.
    text = _RE_HTML_ENTITY.sub(
        lambda m: unicodedata.lookup(
            m.group(0)[1:-1].upper()
        ) if False else _html_entity_decode(m.group(0)),
        text
    )
    text = _RE_HTML_TAG.sub(" ", text)

    # ── 3. URL replacement ────────────────────────────────────────────────────
    # URLs are never sentiment-bearing. The [URL] token is in RoBERTa's Twitter
    # vocabulary (the model was trained with URLs replaced this way), so this
    # substitution is *model-aligned* — not just noise removal.
    text = _RE_URL.sub("[URL]", text)

    # ── 4. Markdown artifact removal ──────────────────────────────────────────
    # Markdown formatting characters become garbage tokens post-tokenization.
    # We preserve the inner text content (see _strip_markdown docstring).
    text, _ = _strip_markdown(text)

    # ── 5. Mod boilerplate stripping ──────────────────────────────────────────
    # Subreddit rule reminders appear identically across thousands of posts
    # and would push the model toward false neutral predictions.
    text = _RE_MOD_BOILERPLATE.sub(" ", text)

    # ── 6. Ticker normalization (optional, default ON) ────────────────────────
    # "$BTC crashed" → "BTC crashed"
    # The $ sign tokenizes as a noise token separate from the ticker symbol.
    # Removing it improves model alignment with its Twitter training distribution
    # where tickers appear as $BTC in tweets — BUT the Cardiff model was trained
    # on raw Twitter text which DID include $BTC. Toggle off if you observe
    # regression on your eval set.
    if normalize_tickers:
        text = _RE_TICKER.sub(r"\1", text)

    # ── 7. Price normalization (optional, default OFF) ────────────────────────
    # Disabled by default: transforms too many surface tokens and may confuse
    # the model's learned representations. Only enable if your downstream
    # trading model needs normalized price figures for feature engineering.
    if normalize_prices:
        text = _RE_PRICE_K.sub(lambda m: str(int(float(m.group(1).replace(",", "")) * 1000)), text)
        text = _RE_PRICE_M.sub(lambda m: str(int(float(m.group(1).replace(",", "")) * 1_000_000)), text)

    # ── 8. Character spam collapse ────────────────────────────────────────────
    # "MOOOOOOON" → "MOOOON" (collapse runs > 4 to exactly 4)
    # We keep 4 because:
    #   - "!!!!" is legitimate punctuation intensity (keep)
    #   - "!!!!!!!!" is spam / noise (collapse)
    # The model was trained on Twitter text which has similar patterns.
    text = _RE_CHAR_REPEAT.sub(r"\1\1\1\1", text)

    # ── 9. Whitespace normalization ───────────────────────────────────────────
    # Collapse all whitespace (tabs, newlines, multiple spaces) to single space.
    # Reddit comment bodies often contain multi-line formatting.
    text = _RE_WHITESPACE.sub(" ", text).strip()

    # ── 10. Zero-signal filter ────────────────────────────────────────────────
    # Only discard if the ENTIRE text is a meaningless filler AND no emotional
    # signal is present (see _is_zero_signal for full guard conditions).
    if _is_zero_signal(text):
        return ""

    # ── 11. Hard length truncation ────────────────────────────────────────────
    # RoBERTa has a 512-token limit. Truncating at char level is a safe proxy.
    # The HuggingFace pipeline's truncation=True handles the final token-level
    # truncation; this char-level truncation is an additional safety net for
    # memory efficiency during batch processing.
    if len(text) > max_length:
        text = text[:max_length].rsplit(" ", 1)[0]  # truncate at word boundary

    return text


# ================================================================================
# HTML ENTITY DECODE HELPER
# ================================================================================

_HTML_ENTITIES: dict[str, str] = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
    "&apos;": "'", "&nbsp;": " ", "&mdash;": "—", "&ndash;": "–",
    "&hellip;": "…", "&rsquo;": "'", "&lsquo;": "'", "&rdquo;": '"', "&ldquo;": '"',
}

def _html_entity_decode(entity: str) -> str:
    """Decode common HTML entities. Falls back to stripping the entity."""
    return _HTML_ENTITIES.get(entity, " ")


# ================================================================================
# FEATURE-EXTRACTING CLEAN (BONUS: extended output for ML feature engineering)
# ================================================================================

def clean_text_with_features(
    text,
    normalize_tickers: bool = True,
    normalize_prices: bool = False,
    max_length: int = 512,
) -> TextFeatures:
    """
    Extended version of clean_text() that also returns auxiliary ML features.

    Use this instead of clean_text() when you want to feed additional signals
    into a downstream trading model alongside the raw sentiment score.

    The returned TextFeatures object contains:
      - cleaned_text: identical to what clean_text() returns
      - is_valid: whether this text should be sent to the model
      - emoji_count: number of emoji characters
      - caps_ratio: fraction of alpha chars that are uppercase
      - punct_intensity: count of !! / ??? clusters
      - spam_score: [0, 1] heuristic spam proxy
      - content_hash: MD5 for exact deduplication
      - simhash: 64-bit SimHash for near-duplicate detection

    Performance note: ~2x slower than clean_text() due to feature extraction.
    Use clean_text() for pure inference pipelines, this for feature engineering.

    Args:
        text: Raw Reddit text
        normalize_tickers: Same as clean_text()
        normalize_prices: Same as clean_text()
        max_length: Same as clean_text()

    Returns:
        TextFeatures dataclass
    """
    # ── Pre-clean feature extraction (on raw text) ────────────────────────────
    raw = str(text) if text is not None else ""

    had_url = bool(_RE_URL.search(raw))
    had_bot = _is_bot_or_removed(raw)
    had_markdown = bool(
        _RE_MARKDOWN_BOLD_ITALIC.search(raw) or
        _RE_MARKDOWN_HEADER.search(raw) or
        _RE_MARKDOWN_CODE_BLOCK.search(raw)
    )
    char_repeat_hits = len(_RE_CHAR_REPEAT.findall(raw))

    # ── Run core cleaner ──────────────────────────────────────────────────────
    cleaned = clean_text(raw, normalize_tickers, normalize_prices, max_length)
    is_valid = len(cleaned) > 0

    if not is_valid:
        return TextFeatures(
            cleaned_text="",
            is_valid=False,
            had_url=had_url,
            had_bot_content=had_bot,
            had_markdown=had_markdown,
            char_repeat_hits=char_repeat_hits,
        )

    # ── Post-clean feature extraction (on cleaned text) ───────────────────────
    emoji_count = _count_emojis(cleaned)
    caps_ratio_val = _caps_ratio(cleaned)
    punct_hits = _RE_PUNCT_INTENSITY.findall(cleaned)
    punct_intensity = len(punct_hits)
    exclamation_count = cleaned.count("!")
    question_count = cleaned.count("?")
    token_count = len(cleaned.split())

    spam = _spam_score(cleaned, char_repeat_hits, token_count)

    content_hash = hashlib.md5(cleaned.encode("utf-8")).hexdigest()
    sim = _simhash(cleaned)

    return TextFeatures(
        cleaned_text=cleaned,
        is_valid=True,
        emoji_count=emoji_count,
        caps_ratio=round(caps_ratio_val, 4),
        punct_intensity=punct_intensity,
        exclamation_count=exclamation_count,
        question_count=question_count,
        char_repeat_hits=char_repeat_hits,
        spam_score=round(spam, 4),
        token_count=token_count,
        content_hash=content_hash,
        simhash=sim,
        had_url=had_url,
        had_bot_content=had_bot,
        had_markdown=had_markdown,
    )


# ================================================================================
# BATCH-SAFE PANDAS INTEGRATION
# ================================================================================

def apply_cleaning_to_df(
    df: pd.DataFrame,
    text_column: str,
    extract_features: bool = False,
    normalize_tickers: bool = True,
    normalize_prices: bool = False,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """
    Apply cleaning to a DataFrame column. Drop-in replacement for the existing
    df["cleaned_text"] = df[col].apply(clean_text) pattern in sentiment_analysis.py.

    Usage in your pipeline (replace the clean_text call in add_sentiment_to_df):

        df = apply_cleaning_to_df(df, text_column="title")
        # or for comments:
        df = apply_cleaning_to_df(df, text_column="comment_text")

    Args:
        df:                 Input DataFrame (not modified in place)
        text_column:        Column containing raw text
        extract_features:   If True, extract auxiliary ML features into new columns
        normalize_tickers:  Pass through to clean_text()
        normalize_prices:   Pass through to clean_text()
        drop_invalid:       If True, drop rows where cleaned text is empty

    Returns:
        pd.DataFrame: Copy with "cleaned_text" column (+ feature columns if requested).
                      Invalid rows are dropped if drop_invalid=True.
    """
    df = df.copy()

    if extract_features:
        # Use the feature-extracting variant
        features_series = df[text_column].apply(
            lambda t: clean_text_with_features(
                t,
                normalize_tickers=normalize_tickers,
                normalize_prices=normalize_prices,
            )
        )
        df["cleaned_text"] = features_series.apply(lambda f: f.cleaned_text)
        df["is_valid"] = features_series.apply(lambda f: f.is_valid)
        df["emoji_count"] = features_series.apply(lambda f: f.emoji_count)
        df["caps_ratio"] = features_series.apply(lambda f: f.caps_ratio)
        df["punct_intensity"] = features_series.apply(lambda f: f.punct_intensity)
        df["exclamation_count"] = features_series.apply(lambda f: f.exclamation_count)
        df["question_count"] = features_series.apply(lambda f: f.question_count)
        df["spam_score"] = features_series.apply(lambda f: f.spam_score)
        df["token_count"] = features_series.apply(lambda f: f.token_count)
        df["content_hash"] = features_series.apply(lambda f: f.content_hash)
        df["simhash"] = features_series.apply(lambda f: f.simhash)
        df["had_url"] = features_series.apply(lambda f: f.had_url)
        df["had_bot_content"] = features_series.apply(lambda f: f.had_bot_content)
    else:
        df["cleaned_text"] = df[text_column].apply(
            lambda t: clean_text(
                t,
                normalize_tickers=normalize_tickers,
                normalize_prices=normalize_prices,
            )
        )
        df["is_valid"] = df["cleaned_text"].str.len() > 0

    if drop_invalid:
        n_before = len(df)
        df = df[df["is_valid"]].copy()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            import logging
            logging.getLogger("reddit_crypto_cleaner").info(
                f"Dropped {n_dropped:,} invalid rows ({100*n_dropped/n_before:.1f}%)"
            )

    return df.reset_index(drop=True)


# ================================================================================
# DEDUPLICATION UTILITIES
# ================================================================================

def deduplicate_exact(df: pd.DataFrame, hash_column: str = "content_hash") -> pd.DataFrame:
    """
    Remove exact duplicate texts using MD5 hash.

    Requires extract_features=True to have been run (or content_hash computed separately).
    Keeps the first occurrence (earliest by DataFrame order, typically chronological
    if you sort by timestamp before calling this).

    Args:
        df: DataFrame with content_hash column
        hash_column: Name of the hash column

    Returns:
        pd.DataFrame: Deduplicated DataFrame
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=[hash_column], keep="first").copy()
    n_after = len(df)
    if n_before != n_after:
        import logging
        logging.getLogger("reddit_crypto_cleaner").info(
            f"Exact dedup: removed {n_before - n_after:,} duplicates "
            f"({100*(n_before-n_after)/n_before:.1f}%)"
        )
    return df.reset_index(drop=True)


def deduplicate_near(
    df: pd.DataFrame,
    simhash_column: str = "simhash",
    hamming_threshold: int = 3,
) -> pd.DataFrame:
    """
    Remove near-duplicate texts using SimHash Hamming distance.

    Two posts are considered near-duplicates if their SimHash fingerprints
    differ in ≤ hamming_threshold bits (default 3).

    This catches copy-paste variants, minor edits, quoted reposts, and
    cross-posted content that would skew hourly sentiment aggregations.

    ⚠️  O(n²) complexity — use on batches ≤ 50k rows or after exact dedup.
        For millions of rows, consider LSH (Locality Sensitive Hashing) instead.

    Args:
        df: DataFrame with simhash column
        simhash_column: Name of SimHash column
        hamming_threshold: Max bit difference to consider near-duplicate (default 3)

    Returns:
        pd.DataFrame: Near-deduplicated DataFrame (keeps first of each near-dup group)
    """
    if len(df) > 50_000:
        import logging
        logging.getLogger("reddit_crypto_cleaner").warning(
            f"Near-dedup called on {len(df):,} rows. "
            "Consider batching or using LSH for datasets this large."
        )

    hashes = df[simhash_column].tolist()
    keep_mask = [True] * len(hashes)

    for i in range(len(hashes)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(hashes)):
            if not keep_mask[j]:
                continue
            if hamming_distance(hashes[i], hashes[j]) <= hamming_threshold:
                keep_mask[j] = False

    result = df[keep_mask].copy()
    n_removed = len(df) - len(result)
    if n_removed > 0:
        import logging
        logging.getLogger("reddit_crypto_cleaner").info(
            f"Near-dedup: removed {n_removed:,} near-duplicates"
        )
    return result.reset_index(drop=True)


# ================================================================================
# INTEGRATION PATCH for sentiment_analysis.py
# ================================================================================
# Replace the existing add_sentiment_to_df() clean step with:
#
#   from reddit_crypto_cleaner import apply_cleaning_to_df
#
#   def add_sentiment_to_df(df, text_column, sentiment_pipeline):
#       df = apply_cleaning_to_df(
#           df,
#           text_column=text_column,
#           extract_features=True,   # ← enable for full feature set
#           drop_invalid=True,
#       )
#       texts = df["cleaned_text"].tolist()
#       ...  (rest of the function unchanged)
#
# The cleaned_text column name is identical to what the existing pipeline expects.
# ================================================================================


# ================================================================================
# QUICK SELF-TEST / EXAMPLES
# ================================================================================

if __name__ == "__main__":
    TEST_CASES = [
        # (input, expected_behavior)
        (
            "[deleted]",
            "SHOULD RETURN EMPTY — deleted post"
        ),
        (
            "I am a bot, and this action was performed automatically. "
            "Please contact the moderators of this subreddit if you have any questions.",
            "SHOULD RETURN EMPTY — bot message"
        ),
        (
            "BTC is going to the MOOOOON 🚀🚀🚀 HODL!!! Not selling!!!",
            "SHOULD PRESERVE — emoji, caps, slang, punct intensity"
        ),
        (
            "Just bought more $BTC at the dip. Diamond 💎 hands baby!!!",
            "SHOULD NORMALIZE TICKER $BTC→BTC, preserve emoji and punct"
        ),
        (
            "**REKT** again lol... https://some-pump-site.com/referral123 check this out",
            "SHOULD STRIP markdown, replace URL with [URL], preserve caps slang"
        ),
        (
            "lol",
            "SHOULD RETURN EMPTY — zero signal"
        ),
        (
            "LOL",
            "SHOULD PRESERVE — caps present → not zero signal"
        ),
        (
            "lol 🔥",
            "SHOULD PRESERVE — emoji present → not zero signal"
        ),
        (
            "not buying this dump, never again. absolutely REKT",
            "SHOULD PRESERVE — negation, slang, caps all intact"
        ),
        (
            "&amp; the market is FUD &lt;br&gt; total panic selling rn !!!",
            "SHOULD DECODE HTML entities, strip HTML tags, preserve intensity"
        ),
        (
            "AAAAAAAAAAAAA bears are back AAAAAAAAAA sell everything NOW",
            "SHOULD COLLAPSE char run >4 → 4, preserve caps and slang"
        ),
        (
            "   \n\n  ok  \n\n  ",
            "SHOULD RETURN EMPTY — zero signal after whitespace normalization"
        ),
        (
            "BTC will hit 100k this cycle. Not financial advice but DYOR ser 🫡",
            "SHOULD PRESERVE slang (DYOR, ser), crypto ref, emoji — strip NFA boilerplate? NO: "
            "here NFA is inline casual usage not standalone boilerplate"
        ),
    ]

    print("=" * 70)
    print("REDDIT CRYPTO CLEANER — SELF-TEST")
    print("=" * 70)

    for raw, note in TEST_CASES:
        result = clean_text(raw)
        feats = clean_text_with_features(raw)
        print(f"\nINPUT   : {raw[:80]!r}")
        print(f"NOTE    : {note}")
        print(f"OUTPUT  : {result!r}")
        print(f"VALID   : {feats.is_valid} | "
              f"EMOJIS: {feats.emoji_count} | "
              f"CAPS%: {feats.caps_ratio:.0%} | "
              f"PUNCT!!: {feats.punct_intensity} | "
              f"SPAM: {feats.spam_score:.2f} | "
              f"TOKENS: {feats.token_count}")
        print("-" * 70)