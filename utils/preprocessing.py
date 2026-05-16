import re
from urllib.parse import urlparse

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


SUSPICIOUS_KEYWORDS = {
    "login",
    "verify",
    "update",
    "account",
    "secure",
    "bank",
    "confirm",
    "password",
    "gift",
    "win",
    "urgent",
}

# Features extractable from a raw URL at inference time (PhiUSIIL schema)
URL_MODEL_FEATURE_COLUMNS = [
    "URLLength", "DomainLength", "IsDomainIP", "CharContinuationRate",
    "TLDLegitimateProb", "URLCharProb", "TLDLength", "NoOfSubDomain",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
]

# Approximate TLD legitimacy probabilities based on global domain distribution
_TLD_LEGIT_PROBS = {
    "com": 0.522907, "org": 0.079963, "net": 0.050207, "edu": 0.032650,
    "gov": 0.028555, "uk": 0.057606, "de": 0.061933, "au": 0.059441,
    "fr": 0.045, "ca": 0.040, "jp": 0.038, "it": 0.035, "br": 0.030,
    "nl": 0.025, "ru": 0.020, "io": 0.015, "co": 0.012, "info": 0.008,
    "biz": 0.005, "me": 0.005, "ng": 0.010, "za": 0.010, "in": 0.020,
}

_stemmer = PorterStemmer()


def ensure_nltk_resources() -> None:
    import nltk

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)


def clean_text(text: str) -> str:
    ensure_nltk_resources()
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    processed = [_stemmer.stem(tok) for tok in tokens if tok not in stop_words and len(tok) > 1]
    return " ".join(processed)


def extract_url_features(url: str) -> pd.DataFrame:
    import math
    from collections import Counter

    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    # Strip www. prefix for domain length calc
    domain_clean = domain.lstrip("www.") if domain.startswith("www.") else domain
    parts = domain_clean.split(".")
    tld = parts[-1] if parts else ""
    subdomain_count = max(0, len(parts) - 2)

    url_len = len(url)
    domain_len = len(domain_clean)
    tld_len = len(tld)
    total = max(url_len, 1)

    # IP address detection
    is_ip = 1 if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", domain) else 0

    # Character type counts
    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)
    equals = url.count("=")
    qmarks = url.count("?")
    amps = url.count("&")
    special = sum(1 for c in url if not c.isalnum() and c not in "/:.-_~#@?=&%+")

    # Percent-encoding obfuscation
    obfuscated = len(re.findall(r"%[0-9a-fA-F]{2}", url))

    # Char continuation rate (longest run of same character / total length)
    max_run = 1
    cur_run = 1
    for i in range(1, len(url)):
        cur_run = cur_run + 1 if url[i] == url[i - 1] else 1
        max_run = max(max_run, cur_run)
    char_continuation = max_run / total

    # URL char probability (normalised entropy — lower = more random = more suspicious)
    counts = Counter(url.lower())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
    url_char_prob = round(min(1.0, entropy / 6.0), 6)

    feature_map = {
        "URLLength": url_len,
        "DomainLength": domain_len,
        "IsDomainIP": is_ip,
        "CharContinuationRate": round(char_continuation, 6),
        "TLDLegitimateProb": _TLD_LEGIT_PROBS.get(tld, 0.001),
        "URLCharProb": url_char_prob,
        "TLDLength": tld_len,
        "NoOfSubDomain": subdomain_count,
        "HasObfuscation": 1 if obfuscated > 0 else 0,
        "NoOfObfuscatedChar": obfuscated,
        "ObfuscationRatio": round(obfuscated / total, 6),
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": round(letters / total, 6),
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": round(digits / total, 6),
        "NoOfEqualsInURL": equals,
        "NoOfQMarkInURL": qmarks,
        "NoOfAmpersandInURL": amps,
        "NoOfOtherSpecialCharsInURL": special,
        "SpacialCharRatioInURL": round(special / total, 6),
        "IsHTTPS": 1 if parsed.scheme.lower() == "https" else 0,
    }

    return pd.DataFrame([feature_map], columns=URL_MODEL_FEATURE_COLUMNS)
