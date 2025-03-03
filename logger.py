import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# 로그 디렉토리 설정
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 로그 파일 설정
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# 로그 포맷 설정
log_formatter = logging.Formatter("[%(levelname)s][%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# 콘솔 핸들러 (화면 출력)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# 파일 핸들러 (파일 저장, 크기 제한 적용)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
file_handler.setFormatter(log_formatter)

# 로거 설정
logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)  # 로그 레벨 설정 (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# 핸들러 추가
logger.addHandler(console_handler)
logger.addHandler(file_handler)
