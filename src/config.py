import os


PORT = 8009
HOST = "0.0.0.0"
PADDLE_IPADDR = os.environ.get("PADDLE_IPADDR", "0.0.0.0")
PADDLE_PORT = os.environ.get("PADDLE_PORT", "9292")

OCR_API = f"http://{HOST}:{PORT}/api/v1/kor-ocr"
PADDLE_API = f"http://{PADDLE_IPADDR}:{PADDLE_PORT}/detection/prediction"

NAME = "korean_ocr"
