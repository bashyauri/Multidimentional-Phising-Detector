import cv2
import numpy as np
from PIL import Image

try:
    from pyzbar.pyzbar import decode as zbar_decode
except Exception:  # pragma: no cover
    zbar_decode = None


def decode_qr_image(file_stream):
    image = Image.open(file_stream).convert("RGB")

    if zbar_decode is not None:
        decoded = zbar_decode(image)
        if decoded:
            return decoded[0].data.decode("utf-8", errors="ignore")

    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(cv_img)
    return data or None
