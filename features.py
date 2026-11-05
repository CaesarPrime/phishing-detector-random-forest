"""
features.py - URL feature extraction and rule-based analysis
Phishing Detector System Using Random Forest (Pepple Christian Boma)

The same functions are used for training (train.py / Kaggle notebook) and for
live detection (app.py), so the model always receives identically computed
features. The URL is analysed as text only; the website is never visited.
"""
import re
from urllib.parse import urlparse

# Order of features fed to the Random Forest model (Table 3.1)
FEATURE_NAMES = [
    "url_length", "num_dots", "num_subdomains", "num_hyphens", "has_at",
    "ip_host", "uses_https", "num_special", "num_digits", "has_keyword",
    "double_slash_path", "is_shortener",
]

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "update", "bank",
    "confirm", "password", "webscr", "ebayisapi", "billing", "suspend",
]

SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc", "rebrand.ly", "s.id",
}

IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")

# Rule-based analyser settings (Table 3.3)
LONG_URL_LIMIT = 75
MAX_SUBDOMAINS = 2
RULE_THRESHOLD = 2


def normalise(url):
    """Strip spaces and add a default scheme so the URL can be parsed."""
    url = (url or "").strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url):
        url = "http://" + url
    return url


def parse(url):
    """Return (normalised_url, parsed, hostname) or raise ValueError."""
    if not url or not url.strip():
        raise ValueError("No URL provided.")
    norm = normalise(url)
    parsed = urlparse(norm)
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("No host name could be found in the input.")
    return norm, parsed, host


def count_subdomains(host):
    if IP_RE.match(host):
        return 0
    labels = [l for l in host.split(".") if l]
    if labels and labels[0] == "www":
        labels = labels[1:]
    # registered domain + top-level domain = last two labels
    return max(len(labels) - 2, 0)


def extract_features(url):
    """Convert a URL into the numerical feature dictionary of Table 3.1."""
    norm, parsed, host = parse(url)
    lower = norm.lower()
    after_scheme = norm.split("://", 1)[1] if "://" in norm else norm
    f = {
        "url_length": len(norm),
        "num_dots": norm.count("."),
        "num_subdomains": count_subdomains(host),
        "num_hyphens": norm.count("-"),
        "has_at": int("@" in norm),
        "ip_host": int(bool(IP_RE.match(host))),
        "uses_https": int(parsed.scheme.lower() == "https"),
        "num_special": sum(norm.count(c) for c in "?=%&~_+!*,;$"),
        "num_digits": sum(ch.isdigit() for ch in norm),
        "has_keyword": int(any(k in lower for k in SUSPICIOUS_KEYWORDS)),
        "double_slash_path": int("//" in after_scheme),
        "is_shortener": int(host in SHORTENERS or host.replace("www.", "") in SHORTENERS),
    }
    return f


def feature_vector(url):
    f = extract_features(url)
    return [f[n] for n in FEATURE_NAMES]


def rule_analyse(url):
    """Apply the heuristic rules of Table 3.3. Returns (score, is_phishing, reasons)."""
    norm, parsed, host = parse(url)
    f = extract_features(url)
    reasons = []
    if f["url_length"] > LONG_URL_LIMIT:
        reasons.append("URL is longer than %d characters" % LONG_URL_LIMIT)
    if f["ip_host"]:
        reasons.append("Host is an IP address")
    if f["has_at"]:
        reasons.append('URL contains "@"')
    if "-" in host:
        reasons.append("Domain name contains a hyphen")
    if f["num_subdomains"] > MAX_SUBDOMAINS:
        reasons.append("More than %d subdomains" % MAX_SUBDOMAINS)
    if f["has_keyword"]:
        reasons.append("Contains a suspicious keyword")
    if not f["uses_https"]:
        reasons.append("Does not use HTTPS")
    if f["is_shortener"]:
        reasons.append("Uses a URL shortening service")
    score = len(reasons)
    return score, score >= RULE_THRESHOLD, reasons
