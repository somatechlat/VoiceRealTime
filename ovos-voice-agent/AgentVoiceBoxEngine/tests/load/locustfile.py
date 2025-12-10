"""Locust Load Test for AgentVoiceBox.

Tests 100+ concurrent connections against the gateway.

Run with:
    locust -f tests/load/locustfile.py --host=http://localhost:25000

Or headless:
    locust -f tests/load/locustfile.py --host=http://localhost:25000 \
           --users=100 --spawn-rate=10 --run-time=60s --headless

Requirements: 7.1, 14.2
"""

import base64
import uuid

from locust import HttpUser, between, events, task

# Sample audio (1 second of silence in WAV format)
SAMPLE_AUDIO_WAV = base64.b64encode(
    b"RIFF"
    + (36 + 32000).to_bytes(4, "little")
    + b"WAVE"
    + b"fmt "
    + (16).to_bytes(4, "little")
    + (1).to_bytes(2, "little")
    + (1).to_bytes(2, "little")
    + (16000).to_bytes(4, "little")
    + (32000).to_bytes(4, "little")
    + (2).to_bytes(2, "little")
    + (16).to_bytes(2, "little")
    + b"data"
    + (32000).to_bytes(4, "little")
    + b"\x00" * 32000
).decode("ascii")


class GatewayUser(HttpUser):
    """Simulates a user connecting to the gateway."""

    wait_time = between(1, 3)

    def on_start(self):
        """Called when a user starts."""
        self.session_id = f"sess_{uuid.uuid4().hex[:16]}"
        self.tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"
        self.api_key = f"avb_test_{uuid.uuid4().hex}"

    @task(10)
    def health_check(self):
        """Check gateway health endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(5)
    def ready_check(self):
        """Check gateway readiness endpoint."""
        with self.client.get("/ready", catch_response=True) as response:
            if response.status_code in [200, 503]:
                response.success()
            else:
                response.failure(f"Ready check failed: {response.status_code}")

    @task(3)
    def metrics_endpoint(self):
        """Check metrics endpoint."""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics failed: {response.status_code}")

    @task(2)
    def list_voices(self):
        """List available TTS voices."""
        with self.client.get(
            "/v1/tts/voices",
            headers={"Authorization": f"Bearer {self.api_key}"},
            catch_response=True,
        ) as response:
            if response.status_code in [200, 401, 403]:
                response.success()
            else:
                response.failure(f"List voices failed: {response.status_code}")

    @task(1)
    def create_session(self):
        """Create a new realtime session."""
        with self.client.post(
            "/v1/realtime/sessions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "ovos-voice-1",
                "voice": "am_onyx",
                "instructions": "You are a helpful assistant.",
            },
            catch_response=True,
        ) as response:
            if response.status_code in [200, 201, 401, 403]:
                response.success()
                if response.status_code in [200, 201]:
                    try:
                        data = response.json()
                        self.session_id = data.get("id", self.session_id)
                    except Exception:
                        pass
            else:
                response.failure(f"Create session failed: {response.status_code}")


class WebSocketUser(HttpUser):
    """Simulates a WebSocket user (using HTTP fallback for load testing)."""

    wait_time = between(0.5, 2)

    def on_start(self):
        """Called when a user starts."""
        self.session_id = f"sess_{uuid.uuid4().hex[:16]}"
        self.tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"
        self.api_key = f"avb_test_{uuid.uuid4().hex}"

    @task(5)
    def simulate_audio_input(self):
        """Simulate sending audio input (via REST endpoint)."""
        # In real scenario, this would be WebSocket
        # Using REST endpoint for load testing
        with self.client.post(
            "/v1/audio/transcriptions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "audio": SAMPLE_AUDIO_WAV,
                "model": "whisper-1",
            },
            catch_response=True,
        ) as response:
            if response.status_code in [200, 401, 403, 404]:
                response.success()
            else:
                response.failure(f"Audio transcription failed: {response.status_code}")

    @task(3)
    def simulate_tts_request(self):
        """Simulate TTS request."""
        with self.client.post(
            "/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "input": "Hello, this is a test message.",
                "voice": "am_onyx",
                "model": "kokoro-1",
            },
            catch_response=True,
        ) as response:
            if response.status_code in [200, 401, 403, 404]:
                response.success()
            else:
                response.failure(f"TTS request failed: {response.status_code}")


class PortalUser(HttpUser):
    """Simulates a portal user."""

    wait_time = between(2, 5)
    host = "http://localhost:25001"  # Portal API

    def on_start(self):
        """Called when a user starts."""
        self.api_key = f"avb_test_{uuid.uuid4().hex}"

    @task(10)
    def portal_health(self):
        """Check portal health."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Portal health failed: {response.status_code}")

    @task(5)
    def get_dashboard(self):
        """Get dashboard data."""
        with self.client.get(
            "/api/v1/dashboard",
            headers={"Authorization": f"Bearer {self.api_key}"},
            catch_response=True,
        ) as response:
            if response.status_code in [200, 401, 403]:
                response.success()
            else:
                response.failure(f"Dashboard failed: {response.status_code}")

    @task(3)
    def get_usage(self):
        """Get usage data."""
        with self.client.get(
            "/api/v1/dashboard/usage",
            headers={"Authorization": f"Bearer {self.api_key}"},
            catch_response=True,
        ) as response:
            if response.status_code in [200, 401, 403]:
                response.success()
            else:
                response.failure(f"Usage failed: {response.status_code}")

    @task(2)
    def list_api_keys(self):
        """List API keys."""
        with self.client.get(
            "/api/v1/api-keys",
            headers={"Authorization": f"Bearer {self.api_key}"},
            catch_response=True,
        ) as response:
            if response.status_code in [200, 401, 403]:
                response.success()
            else:
                response.failure(f"List API keys failed: {response.status_code}")

    @task(1)
    def get_billing_plans(self):
        """Get billing plans."""
        with self.client.get("/api/v1/billing/plans", catch_response=True) as response:
            if response.status_code in [200, 401, 403]:
                response.success()
            else:
                response.failure(f"Billing plans failed: {response.status_code}")


# Event handlers for custom metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Log request metrics."""
    if exception:
        print(f"Request failed: {name} - {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print("=" * 60)
    print("AgentVoiceBox Load Test Starting")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops."""
    print("=" * 60)
    print("AgentVoiceBox Load Test Complete")
    print("=" * 60)
