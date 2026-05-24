import re
import time
from datetime import datetime

from flask import Blueprint, current_app, jsonify, render_template, request
from sqlalchemy import func

from database.db import db
from database.models import PredictionLog
from utils.fusion import weighted_fusion
from utils.model_loader import registry
from utils.preprocessing import clean_text, extract_url_features
from utils.qr_features import build_qr_feature_frame, decode_qr_text_from_bytes


detection_bp = Blueprint("detection", __name__)

KNOWN_LEGITIMATE_DOMAINS = {
    "github.com", "google.com", "microsoft.com", "apple.com", "amazon.com",
    "facebook.com", "twitter.com", "linkedin.com", "youtube.com", "instagram.com",
    "stackoverflow.com", "reddit.com", "wikipedia.org", "github.io", "bitbucket.org",
    "gitlab.com", "heroku.com", "vercel.com", "netlify.com", "aws.amazon.com",
}

KNOWN_SHORTENER_DOMAINS = {
    "bit.ly", "buff.ly", "cutt.ly", "goo.gl", "is.gd", "lnkd.in", "ow.ly",
    "qr.link", "rebrand.ly", "shorturl.at", "t.co", "tiny.cc", "tinyurl.com",
    "trib.al", "urly.it", "v.gd",
}

# Educational and government domains (generally legitimate)
EDUCATIONAL_DOMAIN_SUFFIXES = {
    ".edu", ".ac.uk", ".ac.za", ".ac.ng", ".ac.id", ".ac.in", ".ac.th", 
    ".edu.au", ".edu.br", ".edu.mx", ".edu.sg", ".edu.cn", ".edu.ng",
    ".edu.pk", ".edu.my", ".edu.ph", ".edu.vn", ".edu.bd",
}

GOVERNMENT_DOMAIN_SUFFIXES = {
    ".gov", ".gov.uk", ".gov.au", ".gov.ca", ".gov.in", ".gov.br", ".gov.sg",
}

# Known brands for typosquatting detection
KNOWN_BRANDS = {
    "amazon", "apple", "google", "microsoft", "facebook", "instagram", "twitter",
    "linkedin", "paypal", "ebay", "netflix", "spotify", "github", "adobe", "dropbox",
    "slack", "zoom", "discord", "reddit", "wikipedia", "youtube", "twitch", "steam",
}

# Common phishing domain patterns
PHISHING_DOMAIN_PATTERNS = [
    r"account.?verify", r"confirm.?identity", r"secure.?login", r"verify.?account",
    r"update.?profile", r"confirm.?payment", r"verify.?payment", r"claim.?prize",
    r"reset.?password", r"confirm.?email", r"verify.?email", r"activate.?account",
    r"claim.?bonus", r"free.?airtime", r"airtime.?bonus", r"bonus.?win",
    r"free.?data", r"free.?recharge", r"win.?bonus",
]

URL_SUSPICIOUS_TOKENS = {
    "account", "airtime", "bank", "bonus", "cash", "claim", "data", "free",
    "gift", "giveaway", "login", "password", "prize", "promo", "recharge",
    "reward", "secure", "update", "verify", "voucher", "win", "winner",
}


def _label_from_probability(probability: float):
    label = "Phishing" if probability >= 0.5 else "Legitimate"
    confidence = probability if probability >= 0.5 else 1 - probability
    return label, float(confidence)


def _count_suspicious_url_tokens(raw_url: str) -> int:
    """Count full URL tokens so short words like 'win' do not match 'windows'."""
    return sum(
        bool(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", raw_url))
        for token in URL_SUSPICIOUS_TOKENS
    )


def _is_typosquatting(domain: str) -> bool:
    """Detect if domain mimics known brand with character substitution."""
    domain_part = domain.split('.')[0].lower()
    
    # Common character substitutions: 0→o, 1→i/l, 3→e, 5→s, 7→t
    substitutions = {
        '0': 'o', '1': ['i', 'l'], '3': 'e', '5': 's', '7': 't',
        'i': ['1', '!'], 'l': ['1', '!'], 'o': '0', 'e': '3', 's': '5', 't': '7'
    }
    
    for brand in KNOWN_BRANDS:
        # Direct contains (e.g., "amaz0n" contains pattern of "amazon")
        if brand in domain_part:
            continue
        
        # Check if domain is a typo variant of brand
        brand_clean = brand.replace('0', 'o').replace('1', 'i').replace('3', 'e').replace('5', 's').replace('7', 't')
        domain_clean = domain_part.replace('0', 'o').replace('1', 'i').replace('3', 'e').replace('5', 's').replace('7', 't')
        
        # Calculate similarity (if >80% chars match, it's likely typosquatting)
        if brand_clean in domain_clean or domain_clean in brand_clean:
            if len(set(brand_clean) & set(domain_clean)) / max(len(brand_clean), len(domain_clean)) > 0.7:
                return True
    
    return False


def _has_phishing_domain_pattern(domain: str) -> bool:
    """Detect common phishing domain patterns."""
    domain_lower = domain.lower()
    for pattern in PHISHING_DOMAIN_PATTERNS:
        if re.search(pattern, domain_lower):
            return True
    return False


def _is_known_shortener_domain(domain: str) -> bool:
    return any(domain == shortener or domain.endswith(f".{shortener}") for shortener in KNOWN_SHORTENER_DOMAINS)


def _log_prediction(source_type, input_text, label, confidence, response_time_ms, fusion_used=False):
    row = PredictionLog(
        source_type=source_type,
        input_text=input_text,
        prediction_label=label,
        confidence=confidence,
        response_time_ms=response_time_ms,
        fusion_used=fusion_used,
    )
    db.session.add(row)
    db.session.commit()


def _predict_url(url: str):
    if not url:
        return "Legitimate", 0.0, 0.0

    raw = url.lower()
    features = extract_url_features(url)
    if registry.url_model is not None:
        expected_columns = list(getattr(registry.url_model, "feature_names_in_", features.columns))
        features = features.reindex(columns=expected_columns, fill_value=0)
        prob = float(registry.url_model.predict_proba(features)[0][1])
    else:
        token_hits = _count_suspicious_url_tokens(raw)
        url_has_ip = bool(re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", raw))
        url_has_at = "@" in raw
        prob = min(0.99, 0.18 + token_hits * 0.18 + (0.28 if url_has_ip else 0.0) + (0.16 if url_has_at else 0.0))

    token_hits = _count_suspicious_url_tokens(raw)
    no_https = not raw.startswith("https://")
    url_has_ip = bool(re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", raw))
    url_has_at = "@" in raw

    heuristic_prob = 0.12 + token_hits * 0.16 + (0.18 if no_https else 0.0) + (0.28 if url_has_ip else 0.0) + (0.12 if url_has_at else 0.0)
    prob = max(prob, min(0.99, heuristic_prob))

    # Extract domain for additional checks
    domain_match = re.search(r"(?:https?://)?(?:www\.)?([a-z0-9-]+\.[a-z0-9-.]+)", raw)
    domain = domain_match.group(1) if domain_match else ""
    
    # Check for typosquatting (e.g., "amaz0n" for "amazon")
    if _is_typosquatting(domain):
        prob = max(prob, 0.75)
    
    # Check for common phishing domain patterns (e.g., "account-verify")
    if _has_phishing_domain_pattern(domain):
        prob = max(prob, 0.72)

    # Shortened QR links can hide the real destination from offline analysis.
    if _is_known_shortener_domain(domain):
        prob = max(prob, 0.62)
    
    # Check legitimacy whitelist
    is_known_legitimate = any(domain.endswith(known) or domain == known for known in KNOWN_LEGITIMATE_DOMAINS)
    
    # Check educational/government domains
    is_educational_gov = any(domain.endswith(suffix) for suffix in EDUCATIONAL_DOMAIN_SUFFIXES | GOVERNMENT_DOMAIN_SUFFIXES)

    if is_known_legitimate or is_educational_gov:
        prob = min(prob * 0.15, 0.25)

    label, confidence = _label_from_probability(prob)
    return label, confidence, prob


def _predict_text(text: str, model):
    if not text:
        return "Legitimate", 0.0, 0.0

    if model is not None:
        uses_raw_text = getattr(model, "expects_clean_text", True) is False
        model_input = text if uses_raw_text else clean_text(text)
        prob = float(model.predict_proba([model_input])[0][1])
    else:
        cleaned = clean_text(text)
        suspicious_tokens = ["urgent", "click", "verify", "password", "lottery", "bank"]
        hit_count = sum(tok in cleaned for tok in suspicious_tokens)
        prob = min(1.0, 0.2 + hit_count * 0.15)

    # Rule-based safety boost for common phishing language with links/money claims.
    raw = text.lower()
    has_url = bool(re.search(r"https?://|www\.", raw))
    money_terms = ["prize", "won", "winner", "cash", "claim", "reward", "bonus", "offer", "free", "gift"]
    action_terms = ["now", "urgent", "click", "verify", "limited", "act fast", "confirm"]
    has_money_signal = any(term in raw for term in money_terms)
    action_hits = sum(term in raw for term in action_terms)

    if has_url and has_money_signal:
        prob = max(prob, 0.82)
    elif has_money_signal and action_hits >= 1:
        prob = max(prob, 0.7)

    label, confidence = _label_from_probability(prob)
    return label, confidence, prob


def _predict_qr(file_bytes: bytes):
    decoded_text = decode_qr_text_from_bytes(file_bytes)

    url_prob = None
    if decoded_text:
        _, _, url_prob = _predict_url(decoded_text)

    qr_prob = None
    if registry.qr_model is not None:
        qr_features = build_qr_feature_frame(
            file_bytes=file_bytes,
            url_model=registry.url_model,
            decoded_text=decoded_text,
        )
        qr_prob = float(registry.qr_model.predict_proba(qr_features)[0][1])

    if url_prob is None and qr_prob is None:
        raise RuntimeError("No QR signal available: could not decode QR and no qr_model is loaded.")

    if url_prob is not None and qr_prob is not None:
        final_prob = 0.7 * url_prob + 0.3 * qr_prob
        model_status = "url_qr_fusion"
    elif url_prob is not None:
        final_prob = url_prob
        model_status = "url_model_only"
    else:
        final_prob = qr_prob
        model_status = "qr_model_only"

    label, confidence = _label_from_probability(final_prob)
    return label, confidence, float(final_prob), decoded_text, model_status


def _select_text_model(task: str):
    if task == "sms":
        if current_app.config.get("SMS_PREFER_TRANSFORMER", False):
            return registry.sms_transformer_model or registry.sms_model
        return registry.sms_model or registry.sms_transformer_model

    if task == "email":
        if current_app.config.get("EMAIL_PREFER_TRANSFORMER", True):
            return registry.email_transformer_model or registry.email_model
        return registry.email_model or registry.email_transformer_model

    raise ValueError(f"Unsupported text detection task: {task}")


def _predict_deepfake(file_bytes: bytes, filename: str):
    if registry.deepfake_cnn_model is not None:
        try:
            prob = float(registry.deepfake_cnn_model.predict_probability(file_bytes, filename))
            prob = min(0.99, max(0.01, prob))
            threshold = float(getattr(registry.deepfake_cnn_model, "decision_threshold", 0.5))
            label = "Phishing" if prob >= threshold else "Legitimate"
            confidence = prob if label == "Phishing" else 1 - prob
            architecture = getattr(registry.deepfake_cnn_model, "architecture", "cnn")
            return label, confidence, prob, f"{architecture}_frame_model"
        except Exception as exc:
            raise RuntimeError("Deepfake CNN inference failed for the uploaded media.") from exc

    raise RuntimeError(
        "Deepfake CNN model is not loaded. Train models/deepfake_resnet50.pt or models/deepfake_efficientnet_b0.pt and reload models."
    )


@detection_bp.route("/")
def home():
    return render_template("index.html", current_time=datetime.utcnow())


@detection_bp.route("/api/reload-models", methods=["POST"])
def reload_models():
    registry.load()
    return jsonify({"message": "Models reloaded"})


@detection_bp.route("/api/detect/url", methods=["POST"])
def detect_url():
    start = time.perf_counter()
    payload = request.get_json(silent=True) or request.form
    url = str(payload.get("url", "")).strip()

    label, confidence, prob = _predict_url(url)
    elapsed = (time.perf_counter() - start) * 1000

    _log_prediction("url", url, label, confidence, elapsed)
    return jsonify({"source": "url", "label": label, "confidence": confidence, "phishing_probability": prob})


@detection_bp.route("/api/detect/email", methods=["POST"])
def detect_email():
    start = time.perf_counter()
    payload = request.get_json(silent=True) or request.form
    content = str(payload.get("email_text", "")).strip()

    email_model = _select_text_model("email")
    label, confidence, prob = _predict_text(content, email_model)
    elapsed = (time.perf_counter() - start) * 1000

    _log_prediction("email", content[:1000], label, confidence, elapsed)
    return jsonify({"source": "email", "label": label, "confidence": confidence, "phishing_probability": prob})


@detection_bp.route("/api/detect/sms", methods=["POST"])
def detect_sms():
    start = time.perf_counter()
    payload = request.get_json(silent=True) or request.form
    content = str(payload.get("sms_text", "")).strip()

    sms_model = _select_text_model("sms")
    label, confidence, prob = _predict_text(content, sms_model)
    elapsed = (time.perf_counter() - start) * 1000

    _log_prediction("sms", content[:1000], label, confidence, elapsed)
    return jsonify({"source": "sms", "label": label, "confidence": confidence, "phishing_probability": prob})


@detection_bp.route("/api/detect/qr", methods=["POST"])
def detect_qr():
    start = time.perf_counter()
    if "qr_file" not in request.files:
        return jsonify({"error": "No QR file uploaded"}), 400

    qr_file = request.files["qr_file"]
    file_bytes = qr_file.read()

    try:
        label, confidence, prob, decoded_text, model_status = _predict_qr(file_bytes)
    except RuntimeError as exc:
        return jsonify({"error": str(exc), "model_status": "qr_unavailable"}), 503

    elapsed = (time.perf_counter() - start) * 1000

    _log_prediction("qr", (decoded_text or "")[:1000], label, confidence, elapsed)
    return jsonify(
        {
            "source": "qr",
            "decoded_url": decoded_text,
            "label": label,
            "confidence": confidence,
            "phishing_probability": prob,
            "model_status": model_status,
        }
    )


@detection_bp.route("/api/detect/deepfake", methods=["POST"])
def detect_deepfake():
    start = time.perf_counter()
    if "media_file" not in request.files:
        return jsonify({"error": "No media file uploaded"}), 400

    media_file = request.files["media_file"]
    data = media_file.read()

    try:
        label, confidence, prob, model_status = _predict_deepfake(data, media_file.filename)
    except RuntimeError as exc:
        return jsonify({"error": str(exc), "model_status": "cnn_unavailable"}), 503

    elapsed = (time.perf_counter() - start) * 1000

    _log_prediction("deepfake", media_file.filename or "uploaded_file", label, confidence, elapsed)
    return jsonify(
        {
            "source": "deepfake",
            "label": label,
            "confidence": confidence,
            "phishing_probability": prob,
            "model_status": model_status,
        }
    )


@detection_bp.route("/api/detect/fusion", methods=["POST"])
def detect_fusion():
    start = time.perf_counter()
    payload = request.get_json(silent=True) or {}

    prediction_probs = {}

    if payload.get("url"):
        _, _, prob = _predict_url(str(payload["url"]))
        prediction_probs["url"] = prob

    if payload.get("email_text"):
        email_model = _select_text_model("email")
        _, _, prob = _predict_text(str(payload["email_text"]), email_model)
        prediction_probs["email"] = prob

    if payload.get("sms_text"):
        sms_model = _select_text_model("sms")
        _, _, prob = _predict_text(str(payload["sms_text"]), sms_model)
        prediction_probs["sms"] = prob

    if payload.get("qr_url"):
        _, _, prob = _predict_url(str(payload["qr_url"]))
        prediction_probs["qr"] = prob

    if payload.get("deepfake_probability") is not None:
        prediction_probs["deepfake"] = float(payload["deepfake_probability"])

    final_label, final_prob = weighted_fusion(prediction_probs)
    confidence = final_prob if final_label == "Phishing" else 1 - final_prob

    elapsed = (time.perf_counter() - start) * 1000
    _log_prediction("fusion", str(payload)[:1000], final_label, confidence, elapsed, fusion_used=True)

    return jsonify(
        {
            "source": "fusion",
            "final_label": final_label,
            "confidence": confidence,
            "phishing_probability": final_prob,
            "individual_predictions": prediction_probs,
        }
    )


@detection_bp.route("/api/dashboard-data")
def dashboard_data():
    metrics = registry.metrics or {}

    label_dist_rows = (
        db.session.query(PredictionLog.prediction_label, func.count(PredictionLog.id))
        .group_by(PredictionLog.prediction_label)
        .all()
    )
    label_distribution = {label: count for label, count in label_dist_rows}

    trend_rows = (
        db.session.query(func.date(PredictionLog.created_at), func.count(PredictionLog.id))
        .group_by(func.date(PredictionLog.created_at))
        .order_by(func.date(PredictionLog.created_at))
        .all()
    )
    trends = [{"date": str(day), "count": count} for day, count in trend_rows]

    response_rows = (
        db.session.query(PredictionLog.source_type, func.avg(PredictionLog.response_time_ms))
        .group_by(PredictionLog.source_type)
        .all()
    )
    response_times = [{"source": src, "avg_ms": round(avg_ms or 0, 3)} for src, avg_ms in response_rows]

    recent_logs = (
        db.session.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(10)
        .all()
    )

    model_scores = {
        "models": list(metrics.keys()),
        "accuracy": [metrics[m].get("accuracy", 0) for m in metrics],
        "precision": [metrics[m].get("precision", 0) for m in metrics],
        "recall": [metrics[m].get("recall", 0) for m in metrics],
        "f1": [metrics[m].get("f1", 0) for m in metrics],
        "roc_auc": [metrics[m].get("roc_auc", 0) or 0 for m in metrics],
    }

    confusion = {
        model_name: {
            "tp": model_metrics.get("tp", 0),
            "tn": model_metrics.get("tn", 0),
            "fp": model_metrics.get("fp", 0),
            "fn": model_metrics.get("fn", 0),
        }
        for model_name, model_metrics in metrics.items()
    }

    return jsonify(
        {
            "model_metrics": model_scores,
            "confusion": confusion,
            "label_distribution": label_distribution,
            "trends": trends,
            "response_times": response_times,
            "recent_logs": [row.to_dict() for row in recent_logs],
        }
    )
