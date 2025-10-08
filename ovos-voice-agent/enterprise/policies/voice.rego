package voice

default allow = true

allow {
  input.method == "POST"
  input.path == "/v1/realtime/client_secrets"
}

allow {
  input.method == "POST"
  input.path == "/v1/realtime/sessions"
}

# Extend with fine-grained rules per project_id or metadata as needed.
