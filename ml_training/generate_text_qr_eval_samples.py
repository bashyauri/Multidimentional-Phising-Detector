import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "datasets" / "evaluation_samples"


LEGITIMATE_URLS = [
    "https://www.google.com",
    "https://www.wikipedia.org",
    "https://github.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.amazon.com",
    "https://www.linkedin.com",
    "https://stackoverflow.com",
    "https://www.youtube.com",
    "https://www.reddit.com",
    "https://www.bbc.com",
    "https://www.cnn.com",
    "https://www.coursera.org",
    "https://www.edx.org",
    "https://www.who.int",
    "https://www.gov.uk",
    "https://www.harvard.edu",
    "https://www.ox.ac.uk",
    "https://www.python.org",
    "https://flask.palletsprojects.com",
]

PHISHING_URLS = [
    "http://amaz0n-account-verify.net",
    "http://free-airtime-bonus-win.net",
    "http://paypal-secure-login-alert.com",
    "http://bank-update-profile-now.net",
    "http://microsoft-password-reset-verify.com",
    "http://appleid-confirm-payment.net",
    "http://facebook-security-check-login.com",
    "http://claim-prize-winner-bonus.com",
    "http://secure-login-bank-alert.net",
    "http://verify-account-limited-access.com",
    "http://bonus-data-recharge-free.net",
    "http://netflix-payment-update-now.com",
    "http://instagram-verify-badge-login.net",
    "http://office365-mailbox-quota-alert.com",
    "http://delivery-fee-confirm-payment.net",
    "http://student-loan-forgiveness-claim.com",
    "http://crypto-wallet-verify-seed.com",
    "http://airtime-giveaway-claim-now.net",
    "http://tax-refund-secure-confirm.com",
    "http://qr.link/vEyXM7",
]

LEGITIMATE_SMS = [
    "Hi, I will call you when I get home.",
    "Your appointment is confirmed for tomorrow at 10 AM.",
    "Please send me the lecture notes when you are free.",
    "The meeting has been moved to 2 PM.",
    "Your package has been delivered to reception.",
    "Can you pick up groceries on your way back?",
    "Your bank statement is available in your official mobile app.",
    "Reminder: project supervision meeting starts at noon.",
    "Thank you for attending today's seminar.",
    "The library book you requested is ready for collection.",
    "I am running late but I will be there soon.",
    "Your electricity token purchase was successful.",
    "Please review the attached timetable before class.",
    "The office will be closed on Monday for the holiday.",
    "Your password was changed successfully from your account settings.",
    "The driver is waiting at the main gate.",
    "Payment received. Thank you for your purchase.",
    "Your course registration has been approved.",
    "Lunch is ready. Come downstairs.",
    "The event starts at 6 PM in the auditorium.",
]

PHISHING_SMS = [
    "Congratulations! You won free airtime. Claim now at http://free-airtime-bonus-win.net",
    "Your bank account is locked. Verify immediately at http://bank-update-profile-now.net",
    "You have been selected for a cash reward. Click http://claim-prize-winner-bonus.com",
    "Urgent: confirm your payment details at http://verify-account-limited-access.com",
    "Your PayPal login has expired. Restore access at http://paypal-secure-login-alert.com",
    "Free data bonus available today only. Claim http://bonus-data-recharge-free.net",
    "Netflix billing failed. Update card now at http://netflix-payment-update-now.com",
    "Microsoft account suspended. Reset password at http://microsoft-password-reset-verify.com",
    "Delivery pending. Pay small release fee at http://delivery-fee-confirm-payment.net",
    "Tax refund ready. Confirm identity at http://tax-refund-secure-confirm.com",
    "Apple ID payment issue. Verify now at http://appleid-confirm-payment.net",
    "Instagram badge approved. Login to claim http://instagram-verify-badge-login.net",
    "Crypto wallet alert. Verify seed phrase at http://crypto-wallet-verify-seed.com",
    "Student loan forgiveness approved. Claim today at http://student-loan-forgiveness-claim.com",
    "Office mailbox full. Login to expand quota http://office365-mailbox-quota-alert.com",
    "You won a recharge voucher. Claim now http://airtime-giveaway-claim-now.net",
    "Security alert: unusual activity detected. Verify your account now.",
    "Act fast. Your reward expires in 30 minutes.",
    "Final warning: your account will be closed unless you confirm your password.",
    "Click this secure link now to avoid service suspension.",
]

LEGITIMATE_EMAILS = [
    ("Lecture timetable update", "Dear student, please find attached the revised timetable for next week."),
    ("Meeting notes", "Hello team, attached are the notes from today's project meeting."),
    ("Library notification", "Your requested book is ready for collection at the circulation desk."),
    ("Course registration", "Your course registration has been approved by the department."),
    ("Invoice receipt", "Thank you for your payment. Your official receipt is attached."),
    ("Seminar reminder", "This is a reminder that the research seminar starts at 11 AM tomorrow."),
    ("Password changed", "Your account password was changed successfully from your profile settings."),
    ("Travel itinerary", "Please review the attached itinerary for your approved trip."),
    ("Application received", "We confirm receipt of your application and will respond shortly."),
    ("Maintenance notice", "Scheduled maintenance will occur on Saturday between 1 AM and 3 AM."),
]

PHISHING_EMAILS = [
    ("Account verification required", "Your account has been suspended. Verify your login immediately at http://verify-account-limited-access.com"),
    ("Payment failed", "Your payment could not be processed. Update your card now at http://delivery-fee-confirm-payment.net"),
    ("Prize claim", "Congratulations, you have won a cash prize. Claim now at http://claim-prize-winner-bonus.com"),
    ("Mailbox quota alert", "Your mailbox is full. Login now to avoid deactivation: http://office365-mailbox-quota-alert.com"),
    ("Bank security alert", "Unusual activity detected. Confirm your banking profile at http://bank-update-profile-now.net"),
    ("Password reset", "Your password expires today. Reset immediately at http://microsoft-password-reset-verify.com"),
    ("Refund notice", "A tax refund is waiting. Confirm identity at http://tax-refund-secure-confirm.com"),
    ("Streaming billing issue", "Your subscription will be cancelled unless you update billing at http://netflix-payment-update-now.com"),
    ("Crypto wallet verification", "Protect your wallet by verifying your seed phrase at http://crypto-wallet-verify-seed.com"),
    ("Free airtime bonus", "You qualify for free airtime. Claim at http://free-airtime-bonus-win.net"),
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _generate_qr_images(urls: list[str], output_dir: Path, prefix: str) -> bool:
    try:
        import qrcode
    except ImportError:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, url in enumerate(urls, start=1):
        image = qrcode.make(url)
        image.save(output_dir / f"{prefix}_{index:02d}.png")
    return True


def main() -> None:
    _write_csv(
        OUT_DIR / "url" / "legitimate_urls.csv",
        ["url", "label"],
        [{"url": url, "label": "legitimate"} for url in LEGITIMATE_URLS],
    )
    _write_csv(
        OUT_DIR / "url" / "phishing_urls.csv",
        ["url", "label"],
        [{"url": url, "label": "phishing"} for url in PHISHING_URLS],
    )
    _write_csv(
        OUT_DIR / "sms" / "legitimate_sms_samples.csv",
        ["text", "label"],
        [{"text": text, "label": "ham"} for text in LEGITIMATE_SMS],
    )
    _write_csv(
        OUT_DIR / "sms" / "phishing_sms_samples.csv",
        ["text", "label"],
        [{"text": text, "label": "spam"} for text in PHISHING_SMS],
    )
    _write_csv(
        OUT_DIR / "email" / "legitimate_email_samples.csv",
        ["subject", "body", "label"],
        [{"subject": subject, "body": body, "label": "legitimate"} for subject, body in LEGITIMATE_EMAILS],
    )
    _write_csv(
        OUT_DIR / "email" / "phishing_email_samples.csv",
        ["subject", "body", "label"],
        [{"subject": subject, "body": body, "label": "phishing"} for subject, body in PHISHING_EMAILS],
    )

    qr_ok = True
    qr_ok &= _generate_qr_images(LEGITIMATE_URLS[:10], OUT_DIR / "qr" / "legitimate", "legitimate_qr")
    qr_ok &= _generate_qr_images(PHISHING_URLS[:10], OUT_DIR / "qr" / "phishing", "phishing_qr")

    print(f"Generated URL, SMS, and email evaluation CSVs in {OUT_DIR}")
    if qr_ok:
        print("Generated QR evaluation PNGs.")
    else:
        print("QR PNG generation skipped: install qrcode with 'pip install qrcode[pil]'.")


if __name__ == "__main__":
    main()
