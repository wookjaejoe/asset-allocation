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
log_formatter = logging.Formatter("[%(levelname)s][%(asctime)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

# 콘솔 핸들러 (화면 출력)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.DEBUG)  # ★ 콘솔에서도 DEBUG 레벨까지 출력

# 파일 핸들러 (파일 저장, 크기 제한 적용)
file_handler = RotatingFileHandler(LOG_FILE,
                                   maxBytes=10 * 1024 * 1024,
                                   backupCount=5,
                                   encoding="utf-8")
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)     # ★ 파일에도 DEBUG 레벨까지 출력

# 로거 생성 (이름: app_logger)
logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)  # ★ 로거 자체도 DEBUG 허용

# 기존에 등록된 핸들러와 중복되지 않도록 필요시 제거
if logger.hasHandlers():
    logger.handlers.clear()

# 핸들러 추가
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 테스트
logger.debug("This is a DEBUG log")
logger.info("This is an INFO log")
logger.warning("This is a WARNING log")
logger.error("This is an ERROR log")
logger.critical("This is a CRITICAL log")