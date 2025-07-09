import os
import json
import random
import re
import logging
from flask import Flask, request, render_template, jsonify, redirect, url_for
from datetime import datetime, timedelta
import docx
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from markupsafe import Markup
import openai
from dotenv import load_dotenv
import tiktoken

# Load environment variables from .env
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "medai123")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants from env or defaults
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8000"))
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "25"))

# Paths and folders
MEMORY_PATH = "user_data/talia_memory.json"
UPLOAD_FOLDER = "user_data/docs"
FLASHCARD_PATHS = [
    "user_data/anki_flashcards/COMP_Anki_Deck_Cleaned.json",
    "user_data/anki_flashcards/COMP_Anki_Deck_Merged_With_Salvaged.json",
    "user_data/anki_flashcards/COMP_Anki_Deck.json"
]
VECTOR_DB = "user_data/vector_db"
INDEX_FILE = os.path.join(VECTOR_DB, "comp_questions.index")
TEXTS_FILE = os.path.join(VECTOR_DB, "texts.json")
FLAGGED_QUESTIONS_FILE = "user_data/flagged_questions.json"
ALLOWED_EXTENSIONS = {'txt', 'docx'}

# Ensure folders exist
os.makedirs("user_data", exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB, exist_ok=True)

# SentenceTransformer & FAISS setup
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(384)
question_texts = []
metadata = []

QUOTES = [
    "You don't have to be perfect, just consistent.",
    "One day at a time, one step at a time.",
    "You're not behind. You're on your path.",
    "Progress over perfection.",
    "You've got this, Talia 💪"
]

# Tokenizer for counting tokens (GPT-4)
ENCODER = tiktoken.encoding_for_model("gpt-4")

def count_message_tokens(messages):
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(ENCODER.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # assistant priming tokens
    return num_tokens

def load_all_flashcards():
    card_set = set()
    all_cards = []
    for path in FLASHCARD_PATHS:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for card in json.load(f):
                q, a = card.get("question", "").strip(), card.get("answer", "").strip()
                if q and a and (q, a) not in card_set:
                    all_cards.append({"question": q, "answer": a})
                    card_set.add((q, a))
    return all_cards

ALL_FLASHCARDS = load_all_flashcards()

def sanitize_flashcard(text):
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace(" - ", "\n- ")
    text = re.sub(r'\. (?=[A-Z])', '.<br>', text)
    if len(text) > 500:
        text = text[:500] + "..."
    return Markup(text)

def load_memory():
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r") as f:
            return json.load(f)
    return {
        "messages": [],
        "score": 0,
        "attempts": 0,
        "topics": {},
        "use_comp_mimic": True,
        "setup_complete": False,
        "flashcards_seen": {},
        "comp_date": "",
        "schedule_mode": "schedule",
        "chatbot_name": "Moxie",
        "chatbot_avatar": "robot"
    }

def save_memory(data):
    if "topics" not in data:
        data["topics"] = {}
    with open(MEMORY_PATH, "w") as f:
        json.dump(data, f, indent=2)

def load_flagged_questions():
    if os.path.exists(FLAGGED_QUESTIONS_FILE):
        with open(FLAGGED_QUESTIONS_FILE, "r") as f:
            return json.load(f)
    return []

def save_flagged_questions(flagged_list):
    with open(FLAGGED_QUESTIONS_FILE, "w") as f:
        json.dump(flagged_list, f, indent=2)

def retrieve_similar_questions(prompt, top_k=3):
    if not os.path.exists(INDEX_FILE):
        return []
    faiss_index = faiss.read_index(INDEX_FILE)
    with open(TEXTS_FILE, "r", encoding="utf-8") as f:
        db = json.load(f)
    prompt_vec = model.encode([prompt]).astype('float32')
    D, I = faiss_index.search(prompt_vec, top_k)
    return [db['texts'][i] for i in I[0] if i < len(db['texts'])]

@app.route("/")
def home():
    memory = load_memory()
    if not memory.get("setup_complete", False):
        return redirect(url_for("onboarding_page"))
    quote = random.choice(QUOTES)
    return render_template("index.html", quote=quote)

@app.route("/onboarding")
def onboarding_page():
    return render_template("onboarding.html")

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/get_avatar")
def get_avatar():
    memory = load_memory()
    avatar = memory.get("chatbot_avatar", "robot")
    if avatar not in ("female", "male", "robot"):
        avatar = "robot"
    name = memory.get("chatbot_name", "MediAI")
    return jsonify({"avatar": avatar, "name": name})

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("message", "")
    memory = load_memory()

    flagged_questions = load_flagged_questions()
    flagged_intro = ""
    if flagged_questions:
        flagged_intro = (
            f"\n\nNote: You have {len(flagged_questions)} flagged question(s) "
            "from your quizzes. Feel free to ask me about them or request explanations."
        )

    study_style = memory.get("study_style", "mixed")
    difficulty = memory.get("difficulty", "moderate")
    goals = memory.get("goals", "No goals set")
    learning_style = memory.get("learning_style", "No learning style specified")
    custom_topic = memory.get("custom_topic", "")
    focus_type = memory.get("focus_type", "general")
    chatbot_name = memory.get("chatbot_name", "Moxie")

    topic = request.form.get("topic", custom_topic if custom_topic else "General Principles")

    messages = memory.get("messages", [])
    mimic_enabled = memory.get("use_comp_mimic", False)

    if user_input.startswith("/switch "):
        new_topic = user_input.replace("/switch ", "").strip()
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": f"🔄 Topic switched to **{new_topic}**."})
        memory["last_topic"] = new_topic
        save_memory(memory)
        return jsonify({"response": f"🔄 Topic switched to **{new_topic}**."})

    messages.append({"role": "user", "content": user_input})
    messages = messages[-MAX_MESSAGES:]

    examples = ""
    if mimic_enabled:
        similar_qs = retrieve_similar_questions(user_input)
        examples = "\n\nUse this style as a guide:\n" + "\n\n".join(similar_qs)

    system_prompt = (
        f"You are {chatbot_name}, a caring, intelligent AI tutor guiding Talia through COMP prep. "
        f"Adapt your responses based on her past sessions and preferences.\n"
        f"Talia's study style: {study_style}.\n"
        f"Preferred difficulty: {difficulty}.\n"
        f"Learning style notes: {learning_style}.\n"
        f"Current goals: {goals}.\n"
        f"Focus topic type: {focus_type}.\n"
        f"Today's topic: {topic}."
        f"{flagged_intro}\n"
        f"{examples}"
    )

    messages.insert(0, {"role": "system", "content": system_prompt})

    while count_message_tokens(messages) > MAX_TOKENS and len(messages) > 1:
        messages.pop(1)

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        memory["messages"] = messages
        memory["last_topic"] = topic
        save_memory(memory)

        return jsonify({"response": reply})

    except Exception as e:
        app.logger.error(f"OpenAI API error: {e}")
        return jsonify({"response": "Sorry, I am having trouble processing your request right now. Please try again later."}), 500

@app.route("/clear_flags", methods=["POST"])
def clear_flags():
    save_flagged_questions([])
    return jsonify({"status": "success", "message": "Flagged questions cleared."})

@app.route("/complete_setup", methods=["POST"])
def complete_setup():
    memory = load_memory()
    memory["setup_complete"] = True
    memory["comp_date"] = request.form.get("comp_date", "")
    memory["study_style"] = request.form.get("study_mode", "")
    memory["focus_type"] = request.form.get("topic_mode", "")
    memory["difficulty"] = request.form.get("difficulty", "")
    memory["goals"] = request.form.get("goals", "")
    memory["learning_style"] = request.form.get("learning_style", "")
    memory["custom_topic"] = request.form.get("custom_topic", "").strip()
    chatbot_name = request.form.get("chatbot_name", "").strip()
    memory["chatbot_name"] = chatbot_name if chatbot_name else "Moxie"
    chatbot_avatar = request.form.get("chatbot_avatar", "").strip()
    if chatbot_avatar == "girl":
        memory["chatbot_avatar"] = "female"
    elif chatbot_avatar == "boy":
        memory["chatbot_avatar"] = "male"
    else:
        memory["chatbot_avatar"] = "robot"
    save_memory(memory)
    return redirect(url_for("home"))

@app.route("/flashcards_index")
def flashcards_index():
    memory = load_memory()
    topics = list(memory.get("topics", {}).keys())
    if not topics:
        topics = ["General Principles", "Cardiology", "Neurology", "Endocrinology"]
    return render_template("flashcards_index.html", topics=topics)

@app.route("/flashcard_test")
def flashcard_test():
    topic = request.args.get("topic", None)
    filtered_cards = ALL_FLASHCARDS
    if topic:
        filtered_cards = [card for card in ALL_FLASHCARDS if topic.lower() in card["question"].lower()]
        if not filtered_cards:
            filtered_cards = ALL_FLASHCARDS
    card = random.choice(filtered_cards)
    return render_template("flashcards.html",
                           question=sanitize_flashcard(card["question"]),
                           answer=sanitize_flashcard(card["answer"]))

def update_topic_schedule(topic, accuracy):
    today = datetime.today().strftime("%Y-%m-%d")
    interval = 1 if accuracy <= 50 else 3 if accuracy <= 75 else 5 if accuracy <= 89 else 7
    next_review = (datetime.today() + timedelta(days=interval)).strftime("%Y-%m-%d")
    memory = load_memory()
    topic_data = memory.get("topics", {}).get(topic, {})
    topic_data["last_studied"] = today
    topic_data["next_review"] = next_review
    topic_data["accuracy"] = accuracy
    memory["topics"][topic] = topic_data
    save_memory(memory)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)