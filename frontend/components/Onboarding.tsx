'use client';

import { useState } from 'react';

export default function Onboarding({ onComplete }: { onComplete: () => void }) {
  const [compDate, setCompDate] = useState('');
  const [studyMode, setStudyMode] = useState('flashcards');
  const [topicMode, setTopicMode] = useState('General Principles');
  const [difficulty, setDifficulty] = useState('moderate');
  const [learningStyle, setLearningStyle] = useState('visual');
  const [goals, setGoals] = useState('');

  const submitSetup = async () => {
    await fetch('https://karan20010-8000.app.github.dev/chat', {
      method: 'POST',
      body: new URLSearchParams({
        comp_date: compDate,
        study_mode: studyMode,
        topic_mode: topicMode,
        difficulty,
        goals,
        learning_style: learningStyle,
      }),
    });
    onComplete();
  };

  return (
    <div className="p-4 max-w-xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">🧠 Welcome to MediAI Coach</h1>
      <label className="block mb-2">COMP Exam Date:</label>
      <input className="border p-2 mb-4 w-full" type="date" value={compDate} onChange={(e) => setCompDate(e.target.value)} />

      <label className="block mb-2">Preferred Study Mode:</label>
      <select className="border p-2 mb-4 w-full" value={studyMode} onChange={(e) => setStudyMode(e.target.value)}>
        <option value="flashcards">Flashcards</option>
        <option value="mcq">MCQs</option>
        <option value="vignettes">Vignettes</option>
      </select>

      <label className="block mb-2">Topic Focus:</label>
      <input className="border p-2 mb-4 w-full" value={topicMode} onChange={(e) => setTopicMode(e.target.value)} />

      <label className="block mb-2">Difficulty:</label>
      <select className="border p-2 mb-4 w-full" value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
        <option value="easy">Easy</option>
        <option value="moderate">Moderate</option>
        <option value="hard">Hard</option>
      </select>

      <label className="block mb-2">Learning Style:</label>
      <select className="border p-2 mb-4 w-full" value={learningStyle} onChange={(e) => setLearningStyle(e.target.value)}>
        <option value="visual">Visual</option>
        <option value="written">Written</option>
        <option value="mixed">Mixed</option>
      </select>

      <label className="block mb-2">Your Goals:</label>
      <input className="border p-2 mb-4 w-full" value={goals} onChange={(e) => setGoals(e.target.value)} />

      <button className="bg-blue-600 text-white px-4 py-2 rounded" onClick={submitSetup}>
        Start Studying
      </button>
    </div>
  );
}