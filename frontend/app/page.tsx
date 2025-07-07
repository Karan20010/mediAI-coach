// File: frontend/app/page.tsx
"use client";

import { useState } from "react";
import Onboarding from "../components/Onboarding";

export default function HomePage() {
  const [formData, setFormData] = useState({
    compDate: "",
    studyMode: "Flashcards",
    topicMode: "General Principles",
    difficulty: "Moderate",
    goals: "",
    learningStyle: "Visual",
    customTopic: ""
  });
  const [setupComplete, setSetupComplete] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    await fetch("http://localhost:3000/complete_setup", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({
        comp_date: formData.compDate,
        study_mode: formData.studyMode,
        topic_mode: formData.topicMode,
        difficulty: formData.difficulty,
        goals: formData.goals,
        learning_style: formData.learningStyle,
        custom_topic: formData.customTopic || "",
      }),
    });

    setSetupComplete(true);
  };

  if (setupComplete) {
    return <div className="p-10 text-xl">✅ Setup complete! Ready to begin.</div>;
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6">
      <h1 className="text-2xl font-bold mb-6">🧠 Welcome to MediAI Coach</h1>
      <form onSubmit={handleSubmit} className="w-full max-w-md space-y-4">
        <input
          className="w-full p-2 border rounded"
          type="date"
          value={formData.compDate}
          onChange={(e) => setFormData({ ...formData, compDate: e.target.value })}
          placeholder="yyyy-mm-dd"
          required
        />
        <select
          className="w-full p-2 border rounded"
          value={formData.studyMode}
          onChange={(e) => setFormData({ ...formData, studyMode: e.target.value })}
        >
          <option>Flashcards</option>
          <option>Vignettes</option>
        </select>
        <input
          className="w-full p-2 border rounded"
          type="text"
          value={formData.topicMode}
          onChange={(e) => setFormData({ ...formData, topicMode: e.target.value })}
          placeholder="Topic Focus"
          required
        />
        <select
          className="w-full p-2 border rounded"
          value={formData.difficulty}
          onChange={(e) => setFormData({ ...formData, difficulty: e.target.value })}
        >
          <option>Easy</option>
          <option>Moderate</option>
          <option>Hard</option>
        </select>
        <input
          className="w-full p-2 border rounded"
          type="text"
          value={formData.learningStyle}
          onChange={(e) => setFormData({ ...formData, learningStyle: e.target.value })}
          placeholder="Learning Style (e.g., visual)"
        />
        <input
          className="w-full p-2 border rounded"
          type="text"
          value={formData.goals}
          onChange={(e) => setFormData({ ...formData, goals: e.target.value })}
          placeholder="Your Goals"
        />
        <input
          className="w-full p-2 border rounded"
          type="text"
          value={formData.customTopic}
          onChange={(e) => setFormData({ ...formData, customTopic: e.target.value })}
          placeholder="(Optional) Custom Topic"
        />
        <button
          type="submit"
          className="w-full p-3 bg-blue-600 text-white font-bold rounded hover:bg-blue-700"
        >
          Start Studying
        </button>
      </form>
    </div>
  );
}
