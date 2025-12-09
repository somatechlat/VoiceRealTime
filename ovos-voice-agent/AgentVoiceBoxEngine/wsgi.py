"""WSGI entrypoint for enterprise OVOS voice agent."""

from __future__ import annotations

from app import create_app

# Create the Flask application instance
# Both 'app' and 'application' are provided for compatibility with different WSGI servers
app = create_app()
application = app  # Alias for compatibility

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
