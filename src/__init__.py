from .logs import setup_logger

ocr_logger = setup_logger("KoreanOCR.log", debug=False)

from .coreai.core import ContentExtraction

ocr = ContentExtraction()
