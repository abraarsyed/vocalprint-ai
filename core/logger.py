import logging

logger = logging.getLogger("VocalPrint")
logger.setLevel(logging.INFO)

# Only add handler if not already present (avoid duplicates on reload)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
