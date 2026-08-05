from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    db_username: str
    db_password: str
    db_host: str
    db_port: str
    db_name: str
    data_path: Path
    scaler_path: Path
    model_path: Path

    model_config = SettingsConfigDict(env_file="app/.env")