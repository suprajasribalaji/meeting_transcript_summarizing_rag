import logging
import os
import json
from logging.handlers import RotatingFileHandler

# ===========================
# CREATE LOGS FOLDER
# ===========================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ===========================
# CUSTOM JSON FORMATTER
# ===========================
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        return json.dumps(log_record)

# ===========================
# RETRIEVAL LOGGER
# ===========================
retrieval_logger = logging.getLogger("retrieval")
retrieval_logger.setLevel(logging.INFO)

retrieval_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "retrieval.log"),
    maxBytes=5_000_000,
    backupCount=3
)

retrieval_handler.setFormatter(JsonFormatter())
retrieval_logger.addHandler(retrieval_handler)

# ===========================
# ERROR LOGGER
# ===========================
error_logger = logging.getLogger("error")
error_logger.setLevel(logging.ERROR)

error_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "error.log"),
    maxBytes=5_000_000,
    backupCount=3
)

error_handler.setFormatter(JsonFormatter())
error_logger.addHandler(error_handler)

# ===========================
# CONSOLE LOGGER
# ===========================
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
))

retrieval_logger.addHandler(console_handler)
error_logger.addHandler(console_handler)
