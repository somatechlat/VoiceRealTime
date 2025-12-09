"""Configuration management for the enterprise OVOS voice agent."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field("redis://localhost:6379/0", description="Redis connection URL")
    max_connections: int = Field(200, description="Maximum connections in pool")
    socket_timeout: float = Field(5.0, description="Socket timeout in seconds")
    socket_connect_timeout: float = Field(5.0, description="Connection timeout in seconds")
    retry_on_timeout: bool = Field(True, description="Retry on timeout")
    health_check_interval: int = Field(30, description="Health check interval in seconds")


class KafkaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bootstrap_servers: str = Field(..., description="Kafka bootstrap servers string")
    client_id: str = Field("ovos-voice-agent", description="Kafka client identifier")
    security_protocol: str = Field("PLAINTEXT", description="Kafka security protocol")
    sasl_mechanism: Optional[str] = Field(None, description="Kafka SASL mechanism if needed")
    sasl_username: Optional[str] = Field(None, description="Kafka SASL username")
    sasl_password: Optional[str] = Field(None, description="Kafka SASL password")


class DatabaseSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str = Field(..., description="SQLAlchemy-compatible database URI")
    pool_size: int = Field(10, description="Connection pool size")
    max_overflow: int = Field(20, description="Additional connections allowed above pool size")
    echo: bool = Field(False, description="Enable SQL echo for debugging")


class OPASettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., description="OPA HTTP API base URL")
    decision_path: str = Field("/v1/data/voice/allow", description="Policy decision endpoint")
    timeout_seconds: int = Field(3, description="OPA HTTP request timeout")


class ObservabilitySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_name: str = Field("ovos-voice-agent", description="Service name for tracing/logging")
    log_level: str = Field("INFO", description="Global log level")
    sentry_dsn: Optional[str] = Field(None, description="Optional Sentry DSN for error tracking")
    enable_tracing: bool = Field(False, description="Enable OpenTelemetry tracing export")
    prometheus_namespace: str = Field("ovos", description="Prometheus metrics namespace")


class RateLimitSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests_per_minute: int = Field(60, description="Maximum permitted requests per minute")
    tokens_per_minute: int = Field(120000, description="Maximum permitted tokens per minute")


class SecuritySettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    project_api_keys: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of project identifiers to bearer tokens",
    )
    default_secret_ttl_seconds: int = Field(
        600, description="Default lifetime for realtime client secrets"
    )
    rate_limits: RateLimitSettings = RateLimitSettings()


class AppConfig(BaseSettings):
    """Aggregate configuration pulled from environment variables (.env supported)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )

    flask_env: str = Field("production", alias="FLASK_ENV", description="Flask environment")
    secret_key: str = Field(
        ..., alias="APP_SECRET_KEY", description="Secret key used for Flask session signing"
    )

    redis: RedisSettings = RedisSettings()
    kafka: KafkaSettings
    database: DatabaseSettings
    opa: OPASettings
    security: SecuritySettings = SecuritySettings()
    observability: ObservabilitySettings = ObservabilitySettings()

    def to_flask_config(self) -> Dict[str, Any]:
        return {
            "ENV": self.flask_env,
            "SECRET_KEY": self.secret_key,
        }


def configure_app_from_env() -> AppConfig:
    """Load configuration using the default environment-aware settings class."""

    return AppConfig()


__all__ = [
    "AppConfig",
    "RedisSettings",
    "KafkaSettings",
    "DatabaseSettings",
    "OPASettings",
    "ObservabilitySettings",
    "SecuritySettings",
    "RateLimitSettings",
    "configure_app_from_env",
]
