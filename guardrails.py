"""
PayChat Guardrails — US compliance layer
Runs on every message BEFORE intent actions are triggered.

Covers:
  - PCI DSS: Credit card numbers, CVV, expiry, PIN (MANDATORY for payment apps)
  - Identity theft: SSN, bank account/routing numbers, EIN, ITIN
  - Government IDs: Driver's license, passport numbers
  - BSA / AML: Structuring signals, laundering language, near-$10K amounts
  - Phishing: Typosquat payment URLs, raw IP links, suspicious TLDs
  - Credential leak: Passwords shared in chat
"""

import re
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# PCI DSS — Payment Card Industry Data Security Standard
# Card numbers MUST NOT be stored/processed/transmitted in plaintext.
# Violation: $100K–$500K per incident + loss of payment processing.
# ═══════════════════════════════════════════════════════════════════════

def _luhn_check(num_str: str) -> bool:
    """Validates credit card numbers via Luhn algorithm."""
    digits = [int(d) for d in num_str if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


_CARD_NUMBER_PATTERN = re.compile(
    r'(?:'
    r'4\d{3}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}'           # Visa (16)
    r'|5[1-5]\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}'     # Mastercard
    r'|3[47]\d{2}[\s\-]?\d{6}[\s\-]?\d{5}'                   # Amex (15)
    r'|6(?:011|5\d{2})[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}'  # Discover
    r'|(?:\d{4}[\s\-]){3}\d{4}'                                # Generic 16-digit grouped
    r')'
)

_CVV_PATTERN = re.compile(
    r'\b(?:cvv|cvc|cvv2|cvc2|security\s*code|card\s*code)\s*[:=\-#]?\s*(\d{3,4})\b',
    re.IGNORECASE
)

_EXPIRY_PATTERN = re.compile(
    r'\b(?:exp(?:iry|iration)?|valid\s*(?:thru|through|till|until))\s*[:=\-#]?\s*(\d{2}[\s/\-]\d{2,4})\b',
    re.IGNORECASE
)

_PIN_PATTERN = re.compile(
    r'\b(?:pin|atm\s*pin|card\s*pin|debit\s*pin)\s*[:=\-#]?\s*(\d{4,6})\b',
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════════
# Identity theft — SSN, bank accounts, tax IDs
# ═══════════════════════════════════════════════════════════════════════

_SSN_KEYWORD = re.compile(
    r'\b(?:social\s*security\s*(?:number|num|no|#)?|ssn)\s*[:=\-#]?\s*\d{3}[\s\-]?\d{2}[\s\-]?\d{4}\b',
    re.IGNORECASE
)

_SSN_CONTEXT = re.compile(
    r'\b(?:social\s*security|ssn|tax\s*(?:return|filing|form)|w[\-\s]?2|w[\-\s]?4|i[\-\s]?9|background\s*check|credit\s*(?:check|report|application))\b',
    re.IGNORECASE
)

_SSN_FORMATTED = re.compile(
    r'\b(?!000|666|9\d{2})\d{3}[\s\-]\d{2}[\s\-]\d{4}\b'
)

_ROUTING_NUMBER = re.compile(
    r'\b(?:routing|aba|rtn)\s*(?:number|num|no|#)?\s*[:=\-#]?\s*(\d{9})\b',
    re.IGNORECASE
)

_BANK_ACCOUNT = re.compile(
    r'\b(?:account|acct)\s*(?:number|num|no|#)?\s*[:=\-#]?\s*(\d{8,17})\b',
    re.IGNORECASE
)

_EIN_PATTERN = re.compile(
    r'\b(?:ein|tax\s*id|employer\s*id(?:entification)?)\s*[:=\-#]?\s*\d{2}[\s\-]?\d{7}\b',
    re.IGNORECASE
)

_ITIN_PATTERN = re.compile(
    r'\b(?:itin|individual\s*tax)\s*[:=\-#]?\s*9\d{2}[\s\-]?\d{2}[\s\-]?\d{4}\b',
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════════
# Government IDs — driver's license, passport
# ═══════════════════════════════════════════════════════════════════════

_DRIVERS_LICENSE = re.compile(
    r'\b(?:driver\'?s?\s*licen[sc]e|DL|DMV)\s*(?:number|num|no|#)?\s*[:=\-#]?\s*[A-Z0-9]{6,14}\b',
    re.IGNORECASE
)

_PASSPORT_PATTERN = re.compile(
    r'\b(?:passport)\s*(?:number|num|no|#)?\s*[:=\-#]?\s*[A-Z]?\d{8,9}\b',
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════════
# BSA / AML — Bank Secrecy Act / Anti-Money Laundering
# Financial institutions must report transactions over $10K (CTR)
# Structuring (splitting to avoid $10K threshold) is a federal crime.
# ═══════════════════════════════════════════════════════════════════════

_AML_STRUCTURING = re.compile(
    r'\b(?:split|break|divide)\s+(?:it|the\s+(?:payment|amount|money|transfer))\s+(?:into|to|in)\s+(?:smaller|multiple|separate|parts|pieces)\b',
    re.IGNORECASE
)

_AML_JUST_UNDER_10K = re.compile(
    r'\$\s*(?:9[,.]?[5-9]\d{2}|9[,.]?9\d{2})\b'
)

_AML_LAUNDERING_LANGUAGE = re.compile(
    r'\b(?:clean\s+(?:the\s+)?money|wash\s+(?:the\s+)?money|make\s+it\s+(?:look\s+)?legit(?:imate)?|untraceable|off\s+the\s+books|under\s+the\s+table)\b',
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════════
# Phishing — fake payment URLs, typosquats, suspicious domains
# ═══════════════════════════════════════════════════════════════════════

_PHISHING_DOMAINS = re.compile(
    r'(?:https?://)?(?:www\.)?(?:'
    r'v[e3]nm[o0][\-\.]'                # venmo typosquats
    r'|c[a@]sh[\-\.]?[a@]pp[\-\.]'      # cashapp typosquats
    r'|z[e3]ll[e3][\-\.]'               # zelle typosquats
    r'|p[a@]yp[a@]l[\-\.]'              # paypal typosquats
    r'|[a-z]*(?:verify|secure|login|confirm|update|alert)[\-\.]?(?:venmo|cashapp|zelle|paypal|chase|wellsfargo|bankofamerica|citi)'
    r'|(?:venmo|cashapp|zelle|paypal)[\-\.]?(?:verify|secure|login|confirm|update|support|help|alert)'
    r')[a-z0-9\-]*\.[a-z]{2,}',
    re.IGNORECASE
)

_SUSPICIOUS_URLS = re.compile(
    r'(?:https?://)?(?:'
    r'(?:\d{1,3}\.){3}\d{1,3}[:/]'      # Raw IP addresses
    r'|[a-z0-9\-]+\.(?:tk|ml|ga|cf|gq|xyz|top|club|buzz|icu|pw|cc)\b'  # Free/spam TLDs
    r')',
    re.IGNORECASE
)

_URL_SHORTENERS = re.compile(
    r'\b(?:bit\.ly|tinyurl\.com|t\.co|rb\.gy|is\.gd|v\.gd|shorturl\.at|cutt\.ly)/\S+',
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════════
# Credential / password leak
# ═══════════════════════════════════════════════════════════════════════

_PASSWORD_PATTERN = re.compile(
    r'\b(?:password|passcode|pwd|passwd|login)\s*[:=\-#]?\s*\S{4,}',
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════════
# Main guardrail function
# ═══════════════════════════════════════════════════════════════════════

def run_guardrails(text: str, intents: list) -> Optional[dict]:
    """
    Run all guardrail checks on a message.

    Args:
        text: Raw message text
        intents: List of detected intents (used for context-aware checks)

    Returns:
        None if no issues detected, otherwise EXACTLY these three keys:
        {
            "flags": ["card_number", ...],
            "warnings": ["Human-readable warning message", ...],
            "blocked": True/False (whether to suppress all intents),
        }

        This docstring used to list a fourth key, "scam_type", which the function has
        never returned in any version. The backend modelled its struct on this
        docstring, so Android received {"flags": [], "scam_type": [], "warnings": []}
        and rendered an empty compliance banner. Keep this list matching the return
        statement — it is the only place the response shape is written down.
    """
    flags = []
    warnings = []
    block_intents = False

    # ── PCI DSS: Card numbers (Luhn-validated) ──
    for match in _CARD_NUMBER_PATTERN.finditer(text):
        digits = re.sub(r'\D', '', match.group())
        if _luhn_check(digits):
            flags.append("card_number")
            warnings.append(
                "Credit/debit card number detected — never share card details in chat. "
                "No payment action will be triggered."
            )
            block_intents = True
            break

    if _CVV_PATTERN.search(text):
        flags.append("cvv")
        warnings.append("Card security code (CVV) detected — do not share this in chat.")
        block_intents = True

    if _EXPIRY_PATTERN.search(text):
        flags.append("card_expiry")
        # Every flag must carry warning text. This one and near_reporting_threshold
        # did not, so the payload came back as {"flags": [...], "warnings": []} and the
        # Android client — which shows the compliance banner whenever guardrails is
        # non-null — rendered an empty message.
        warnings.append("Card expiry date detected — avoid sharing card details in chat.")
        if "card_number" in flags or "cvv" in flags:
            block_intents = True

    if _PIN_PATTERN.search(text):
        flags.append("pin")
        warnings.append("PIN detected — never share your PIN with anyone, including in chat.")
        block_intents = True

    # ── SSN (keyword-gated to avoid flagging phone numbers) ──
    ssn_by_keyword = _SSN_KEYWORD.search(text)
    ssn_by_format = _SSN_FORMATTED.search(text) and _SSN_CONTEXT.search(text)
    if ssn_by_keyword or ssn_by_format:
        flags.append("ssn")
        warnings.append(
            "Social Security Number detected — sharing your SSN in chat is extremely dangerous. "
            "No payment action will be triggered."
        )
        block_intents = True

    # ── Bank account / routing ──
    if _ROUTING_NUMBER.search(text):
        flags.append("routing_number")
        warnings.append("Bank routing number detected — do not share banking details in chat.")
        block_intents = True

    if _BANK_ACCOUNT.search(text):
        flags.append("bank_account")
        warnings.append("Bank account number detected — do not share banking details in chat.")
        block_intents = True

    # ── Tax IDs ──
    if _EIN_PATTERN.search(text):
        flags.append("ein")
        warnings.append("Tax ID (EIN) detected — be cautious sharing tax information in chat.")

    if _ITIN_PATTERN.search(text):
        flags.append("itin")
        warnings.append("Individual Tax ID (ITIN) detected — do not share tax IDs in chat.")
        block_intents = True

    # ── Government IDs ──
    if _DRIVERS_LICENSE.search(text):
        flags.append("drivers_license")
        warnings.append("Driver's license number detected — avoid sharing government ID in chat.")

    if _PASSPORT_PATTERN.search(text):
        flags.append("passport")
        warnings.append("Passport number detected — avoid sharing travel documents in chat.")

    # ── AML signals ──
    if _AML_STRUCTURING.search(text):
        flags.append("structuring_signal")
        warnings.append(
            "Splitting payments to avoid reporting thresholds is a federal crime under 31 USC §5324."
        )

    if _AML_LAUNDERING_LANGUAGE.search(text):
        flags.append("laundering_language")
        warnings.append("This message contains language associated with money laundering.")
        block_intents = True

    if "money" in intents and _AML_JUST_UNDER_10K.search(text):
        flags.append("near_reporting_threshold")
        warnings.append(
            "This amount is just under a reporting threshold. Splitting payments to "
            "stay below reporting limits is illegal."
        )

    # ── Phishing URLs ──
    if _PHISHING_DOMAINS.search(text):
        flags.append("phishing_url")
        warnings.append(
            "Suspicious URL detected that may be impersonating a payment service. "
            "Do not click or enter any credentials."
        )
        block_intents = True

    if _SUSPICIOUS_URLS.search(text) and any(i in intents for i in ["money", "bills"]):
        flags.append("suspicious_url")
        warnings.append(
            "Suspicious URL detected alongside a payment request — verify the link before proceeding."
        )

    if _URL_SHORTENERS.search(text) and any(i in intents for i in ["money", "bills"]):
        flags.append("shortened_url")
        warnings.append("Shortened URL detected with a payment request — shortened links can hide malicious destinations.")

    # ── Credential leak ──
    if _PASSWORD_PATTERN.search(text):
        flags.append("password_leak")
        warnings.append("Password or login credential detected in chat — never share passwords in messages.")

    # ── Return ──
    if not flags:
        return None

    # Never hand back a flag with nothing to show for it. Both clients treat a non-null
    # guardrails object as "display the compliance banner", so an empty warnings list
    # renders a blank banner — which is what Android was reporting. If a future flag is
    # added without warning text this keeps the payload honest rather than silently
    # shipping an empty message.
    if not warnings:
        return None

    return {
        "flags": flags,
        "warnings": warnings,
        "blocked": block_intents,
    }
