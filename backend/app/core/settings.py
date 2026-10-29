from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Synthetic Image Generation API"
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"

    KOHYA_DIR: Path = Path(__file__).resolve().parents[3] / "kohya_ss"

    STORAGE_ROOT: Path = (
        Path(__file__).resolve().parents[3] / "storage"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()