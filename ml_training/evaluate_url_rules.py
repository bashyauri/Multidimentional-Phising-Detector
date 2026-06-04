from routes.detection import _predict_url


BENIGN_URLS = [
    "https://www.microsoft.com/en-us/security",
    "https://www.google.com",
    "https://openai.com/blog",
    "https://www.nasa.gov",
    "https://www.un.org",
    "https://www.bbc.com/news",
    "https://www.stanford.edu",
    "https://www.harvard.edu",
    "https://support.zoom.us",
    "https://developer.mozilla.org",
    "https://docs.python.org/3/",
    "https://www.paypal.com/signin",
    "https://accounts.google.com",
    "https://bankofamerica.com",
    "https://chase.com",
    "https://www.nytimes.com",
    "https://www.linkedin.com/feed/",
    "https://www.amazon.com/gp/help/customer/display.html",
    "https://www.apple.com/support/",
    "https://github.com/settings/security",
]

PHISHING_URLS = [
    "http://secure-login-paypal.com/verify",
    "http://fakebank-login.com",
    "http://paypal-account-verify-security.com",
    "http://microsoft-support-verify.net/login",
    "http://goog1e-account-recovery.com",
    "http://appleid-confirm-billing.com",
    "http://bank-alert-update-account.com",
    "http://verify-payments-now.com",
    "http://192.168.1.44/secure/login",
    "http://bit.ly/free-airtime-bonus",
    "http://secure-update-login.net",
    "http://confirm-email-prize.com",
    "http://unlock-account-fast.com",
    "http://winner-cash-claim-now.com",
    "http://tinyurl.com/freegiftverify",
    "http://signin-paypal-security-check.com",
    "http://account-verify-update-security.com",
    "http://amazon-login-confirm-now.com",
    "http://netflix-billing-update-alert.com",
    "http://bank-secure-verify-login.net",
]


def _run_bucket(urls: list[str], expected: str) -> tuple[int, list[tuple[str, str, float]]]:
    correct = 0
    failures = []
    for url in urls:
        label, _conf, prob = _predict_url(url)
        if label.lower() == expected:
            correct += 1
        else:
            failures.append((url, label, float(prob)))
    return correct, failures


def main() -> int:
    benign_ok, benign_fail = _run_bucket(BENIGN_URLS, "legitimate")
    phish_ok, phish_fail = _run_bucket(PHISHING_URLS, "phishing")

    total = len(BENIGN_URLS) + len(PHISHING_URLS)
    correct = benign_ok + phish_ok
    accuracy = correct / total if total else 0.0

    print(f"Benign correct: {benign_ok}/{len(BENIGN_URLS)}")
    print(f"Phishing correct: {phish_ok}/{len(PHISHING_URLS)}")
    print(f"Overall accuracy: {accuracy:.3f}")

    if benign_fail:
        print("\nFalse positives (benign flagged as phishing):")
        for url, label, prob in benign_fail:
            print(f"- {url} => {label} ({prob:.3f})")

    if phish_fail:
        print("\nFalse negatives (phishing flagged as legitimate):")
        for url, label, prob in phish_fail:
            print(f"- {url} => {label} ({prob:.3f})")

    return 0 if not benign_fail and not phish_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
