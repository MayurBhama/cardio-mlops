import logging
import os
import sys
from datetime import datetime

# Create logs directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),  # Write to log file
        logging.StreamHandler()              # Print to console
    ]
)

logger = logging.getLogger(__name__)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception
