# 🚀 QUICK START - Simple OVOS Voice Chat

**No LLM API keys needed!** This uses local OVOS processing.

## 📋 Quick Setup (2 minutes)

### 1. Install Dependencies
```bash
cd ovos-voice-agent/sprint4-websocket
pip install fastapi uvicorn websockets pydantic
```

### 2. Start the Server
```bash
python realtime_server.py
```

### 3. Open the Chat Interface
Open `simple-chat.html` in your browser (double-click the file)

### 4. Start Chatting!
1. Click "Connect to OVOS" 
2. Click the 🎤 button
3. **Start talking** (say something like "Hello, how are you?")
4. Click 🎤 again to stop and get a response

## 🎯 What It Does

- **No API Keys Required** - Uses local OVOS processing
- **Real-time Voice Chat** - Click button, talk, get responses  
- **OpenAI Compatible** - Same protocol as OpenAI's voice API
- **Simple Responses** - Basic conversational AI (expandable with LLM integration)

## 🔧 How It Works

The system provides basic conversational responses:
- "Hello" → "Hello! How can I help you today?"
- "How are you" → "I'm doing well, thank you for asking!"  
- "What time" → Current time
- Other inputs → "I heard you say: [your words]. How can I help you with that?"

## 🚀 Next Steps

To add real LLM intelligence:
1. Install `ovos-persona-server`
2. Connect to local Ollama or remote LLM APIs
3. The system is designed to integrate seamlessly

## 🛠️ Troubleshooting

**"Connection failed"** → Make sure the server is running on port 8001
**"Microphone denied"** → Allow microphone access in browser
**No response** → Check browser console for errors

---

**🎉 You now have a working OpenAI-compatible voice agent running locally!**