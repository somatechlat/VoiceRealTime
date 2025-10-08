"""Kafka client helpers wrapping confluent-kafka for producers/consumers."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from confluent_kafka import Consumer, Producer

from ..config import AppConfig

logger = logging.getLogger(__name__)


class KafkaFactory:
    """Factory that builds Kafka producers and consumers using shared configuration."""

    def __init__(self, config: AppConfig):
        self._config = config

    def _base_config(self) -> Dict[str, Any]:
        base = {
            "bootstrap.servers": self._config.kafka.bootstrap_servers,
            "client.id": self._config.kafka.client_id,
            "security.protocol": self._config.kafka.security_protocol,
        }
        if self._config.kafka.sasl_mechanism:
            base.update(
                {
                    "sasl.mechanisms": self._config.kafka.sasl_mechanism,
                    "sasl.username": self._config.kafka.sasl_username,
                    "sasl.password": self._config.kafka.sasl_password,
                }
            )
        return base

    def create_producer(self, overrides: Optional[Dict[str, Any]] = None) -> Producer:
        cfg = self._base_config()
        if overrides:
            cfg.update(overrides)
        return Producer(cfg)

    def create_consumer(
        self, group_id: str, overrides: Optional[Dict[str, Any]] = None
    ) -> Consumer:
        cfg = self._base_config()
        cfg.update(
            {
                "group.id": group_id,
                "enable.auto.commit": False,
                "auto.offset.reset": "earliest",
            }
        )
        if overrides:
            cfg.update(overrides)
        return Consumer(cfg)


@contextmanager
def kafka_producer(factory: KafkaFactory, **overrides: Any) -> Iterator[Producer]:
    producer = factory.create_producer(overrides)
    try:
        yield producer
    finally:
        producer.flush()


__all__ = ["KafkaFactory", "kafka_producer"]
