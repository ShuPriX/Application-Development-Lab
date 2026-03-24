from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    TMDB_API_KEY: str = ""
    DATABASE_URL: str = "sqlite+aiosqlite:///./cineinsight.db"
    BERT_MODEL_PATH: str = "ml/models/bert_sentiment"
    BILSTM_MODEL_PATH: str = "ml/models/bilstm_aspect"
    SCRAPE_DELAY: float = 1.5
    MAX_REVIEWS_PER_SOURCE: int = 50
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
