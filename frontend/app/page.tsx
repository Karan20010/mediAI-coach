'use client';

import '../styles/globals.css'; // ✅ TailwindCSS styles
import { useState } from 'react';

export default function HomePage() {
  const [messages, setMessages] = useState<string[]>([]);
  const [input, setInput] = useState('');

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessages = [...messages, `🧑‍💻 You: ${input}`];
    setMessages(newMessages);
    setInput('');

    try {
      const res = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input })
      });
      const data = await res.json();
      setMessages([...newMessages, `🤖 MediAI: ${data.reply}`]);
    } catch (err) {
      setMessages([...newMessages, '❌ Error reaching backend.']);
    }
  };

  return (
    <main className="flex flex-col items-center justify-center min-h-screen p-4 bg-gray-100 text-black">
      <h1 className="text-3xl font-bold mb-4">🧠 MediAI Coach</h1>

      <div className="w-full max-w-xl bg-white rounded shadow p-4 h-[400px] overflow-y-auto mb-4">
        {messages.map((msg, idx) => (
          <div key={idx} className="mb-2">{msg}</div>
        ))}
      </div>

      <div className="w-full max-w-xl flex space-x-2">
        <input
          className="flex-grow p-2 border rounded"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Ask me anything..."
        />
        <button className="bg-blue-600 text-white px-4 py-2 rounded" onClick={sendMessage}>
          Send
        </button>
      </div>
    </main>
  );
}