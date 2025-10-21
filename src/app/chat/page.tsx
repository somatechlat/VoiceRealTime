"use client";

import React, { useState, useRef, useEffect } from "react";

/**
 * Minimal UI for interacting with the OVOS realtime voice agent.
 *
 * - Click **Start Session** to open a WebSocket connection.
 * - Type a message and press **Send** – the message is sent as a
 *   `conversation.item.create` event (text based, not audio).
 * - The UI then requests a response via `response.create` and displays the
 *   assistant reply when the `response.done` event arrives.
 *
 * This UI lives under `/chat` in the Next.js app and requires the server to be
 * reachable at `ws://localhost:65000/v1/realtime` (the Docker compose stack).
 */

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export default function ChatPage() {
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const sessionIdRef = useRef<string>("");

  // Open WebSocket when the user clicks the button
  const startSession = () => {
    if (ws) return; // already open
    const socket = new WebSocket("ws://localhost:65000/v1/realtime");
    socket.onopen = () => {
      console.log("WebSocket opened");
    };
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      // Capture the session id from the first session.created event
      if (data.type === "session.created" && data.session?.id) {
        sessionIdRef.current = data.session.id;
        setConnected(true);
        console.log("Session created", data.session.id);
        return;
      }
      // Assistant response finished – display the transcript
      if (data.type === "response.done" && data.response?.output) {
        const assistantMsg = data.response.output[0];
        const text = assistantMsg?.content?.[0]?.transcript ?? "";
        setMessages((prev) => [...prev, { role: "assistant", content: text }]);
      }
    };
    socket.onerror = (e) => console.error("WebSocket error", e);
    socket.onclose = () => {
      console.log("WebSocket closed");
      setConnected(false);
      setWs(null);
    };
    setWs(socket);
  };

  const sendMessage = () => {
    if (!ws || !connected) return;
    const userMsg = input.trim();
    if (!userMsg) return;

    // Show user message locally
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);

    // Send a conversation.item.create event (text based)
    const itemEvent = {
      type: "conversation.item.create",
      item: {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: userMsg }],
      },
    };
    ws.send(JSON.stringify(itemEvent));

    // Ask the assistant to respond
    ws.send(JSON.stringify({ type: "response.create" }));

    setInput("");
  };

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      ws?.close();
    };
  }, [ws]);

  return (
    <div style={{ maxWidth: "600px", margin: "2rem auto", fontFamily: "sans-serif" }}>
      <h1>OVOS Voice Agent Chat</h1>
      {!connected && (
        <button onClick={startSession} style={{ padding: "0.5rem 1rem" }}>
          Start Session
        </button>
      )}
      {connected && (
        <div>
          <div
            style={{
              border: "1px solid #ccc",
              padding: "1rem",
              minHeight: "300px",
              marginBottom: "1rem",
              overflowY: "auto",
            }}
          >
            {messages.map((msg, idx) => (
              <div key={idx} style={{ marginBottom: "0.5rem" }}>
                <strong>{msg.role === "user" ? "You" : "Assistant"}:</strong> {msg.content}
              </div>
            ))}
          </div>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
              placeholder="Type a message…"
              style={{ flex: 1, padding: "0.5rem" }}
            />
            <button onClick={sendMessage} style={{ padding: "0.5rem 1rem" }}>
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
