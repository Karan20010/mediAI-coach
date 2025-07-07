import json
import os

MEMORY_PATH = "user_data/talia_memory.json"

blank_memory = {
    "onboarded": False,
    "messages": [],
    "topics": {},
    "score": 0,
    "attempts": 0,
    "use_comp_mimic": True,
    "setup_complete": False,
    "comp_date": "",
    "study_style": "mixed",
    "focus_type": "general",
    "difficulty": "moderate",
    "goals": "",
    "learning_style": "",
    "custom_topic": "",
    "last_topic": "",
    "chatbot_name": "Moxie",
    "chatbot_avatar": "robot",
    "flashcards_seen": {}
}

os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)

with open(MEMORY_PATH, "w") as f:
    json.dump(blank_memory, f, indent=2)

print(f"Blank memory file created/reset at {MEMORY_PATH}")