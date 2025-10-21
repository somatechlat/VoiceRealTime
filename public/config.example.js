// Client-side OVOS config example. Copy to public/config.js and fill with your values.
window.OVOS_CONFIG = {
  VOICE_AGENT_HOST: window.location.hostname || 'localhost',
  VOICE_AGENT_PORT: 60200,
  VOICE_AGENT_BASE: `http://${window.location.hostname || 'localhost'}:60200`,
  VOICE_AGENT_WS_BASE: `ws://${window.location.hostname || 'localhost'}:60200`,
};
