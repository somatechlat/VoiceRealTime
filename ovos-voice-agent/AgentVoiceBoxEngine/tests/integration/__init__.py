"""Integration tests for AgentVoiceBox.

These tests run against real infrastructure (Redis, PostgreSQL) via Docker Compose.
NO MOCKS, NO FAKES, NO STUBS.

Run with:
    docker compose -f docker-compose.test.yml up -d
    pytest tests/integration/ -v
"""
