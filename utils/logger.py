# utils/logger.py — Clean logging setup using loguru

import sys
import os
from loguru import logger

# Remove default loguru handler
logger.remove()

# Console handler — clean and coloured
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="DEBUG",
    colorize=True
)

# File handler — create logs/ folder automatically
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/assistant.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}"
)