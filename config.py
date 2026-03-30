from pathlib import Path

# --- пути ---
BASE_DIR = Path(__file__).resolve().parent
DROPBOX_DIR = BASE_DIR / "dropbox"
IMAGE_STORE_DIR = BASE_DIR / "image_store"
ARCHIVE_SUCCESS_DIR = BASE_DIR / "archive" / "success"
ARCHIVE_FAILED_DIR = BASE_DIR / "archive" / "failed"

# --- настройки redis/celery ---
REDIS_URL = "redis://localhost:6379/0"

# --- настройки парсера ---
SPACY_MODEL_EN = "en_core_web_trf" # или ru_core_news_sm для русского
SPACY_MODEL_RU = "ru_core_news_lg"