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
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from markupsafe import Markup
import openai
from dotenv import load_dotenv
import tiktoken

load_dotenv()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)

# --- App Setup ---
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "secret123")
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Constants ---
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8000"))
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "25"))

# --- Paths ---
MEMORY_PATH = "user_data/talia_memory.json"
UPLOAD_FOLDER = "user_data/docs"
FLASHCARD_PATHS = [
    "user_data/anki_flashcards/COMP_Anki_Deck.json"
]
VECTOR_DB = "user_data/vector_db"
INDEX_FILE = os.path.join(VECTOR_DB, "comp_questions.index")
TEXTS_FILE = os.path.join(VECTOR_DB, "texts.json")
ALLOWED_EXTENSIONS = {'txt', 'docx'}

# --- Setup ---
os.makedirs("user_data", exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB, exist_ok=True)

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

# --- Tokenizer Setup ---
ENCODER = tiktoken.encoding_for_model("gpt-4")

def count_message_tokens(messages):
    """Count tokens in messages for GPT-4 model accurately."""
    tokens_per_message = 3  # per OpenAI specs
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

# --- Load Flashcards ---
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

# --- Sanitize Flashcard Text ---
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

# --- Utilities ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_memory():
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r") as f:
            return json.load(f)
    return {
        "messages": [], "score": 0, "attempts": 0, "topics": {},
        "use_comp_mimic": True, "setup_complete": False,
        "flashcards_seen": {}, "comp_date": "", "schedule_mode": "schedule",
        "chatbot_name": "Moxie",
        "chatbot_avatar": "robot"  # default avatar
    }

def save_memory(data):
    if "topics" not in data:
        data["topics"] = {}
    with open(MEMORY_PATH, "w") as f:
        json.dump(data, f, indent=2)

def extract_questions_from_docx(path):
    doc = docx.Document(path)
    blocks = []
    current_q = ""
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text: 
            continue
        if text.startswith("Q") or text.endswith("?"):
            if current_q: 
                blocks.append(current_q.strip())
            current_q = text
        else:
            current_q += "\n" + text
    if current_q: 
        blocks.append(current_q.strip())
    return blocks

def rebuild_vector_db():
    global index, question_texts, metadata
    index = faiss.IndexFlatL2(384)
    question_texts = []
    metadata = []
    for file in os.listdir(UPLOAD_FOLDER):
        if file.endswith(".docx"):
            filepath = os.path.join(UPLOAD_FOLDER, file)
            blocks = extract_questions_from_docx(filepath)
            embeddings = model.encode(blocks)
            index.add(np.array(embeddings).astype('float32'))
            question_texts.extend(blocks)
            metadata.extend([{ "source": file }] * len(blocks))
    faiss.write_index(index, INDEX_FILE)
    with open(TEXTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"texts": question_texts, "meta": metadata}, f, indent=2)

def retrieve_similar_questions(prompt, top_k=3):
    if not os.path.exists(INDEX_FILE): 
        return []
    faiss_index = faiss.read_index(INDEX_FILE)
    with open(TEXTS_FILE, "r", encoding="utf-8") as f:
        db = json.load(f)
    prompt_vec = model.encode([prompt]).astype('float32')
    D, I = faiss_index.search(prompt_vec, top_k)
    return [db['texts'][i] for i in I[0] if i < len(db['texts'])]

# --- Routes ---
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

@app.route("/flashcards_index")
def flashcards_index():
    memory = load_memory()
    # For simplicity, use the keys of memory["topics"] as topics if available
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
            filtered_cards = ALL_FLASHCARDS  # fallback if no cards in topic
    card = random.choice(filtered_cards)
    return render_template("flashcards.html",
                           question=sanitize_flashcard(card["question"]),
                           answer=sanitize_flashcard(card["answer"]))

@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    memory = load_memory()
    user_answer = request.form["answer"].strip().upper()
    correct = request.form["correct"].strip().upper()
    memory["attempts"] += 1
    topic = memory.get("last_topic", "General Principles")
    memory["flashcards_seen"][topic] = memory.get("flashcards_seen", {}).get(topic, 0) + 1
    if user_answer == correct:
        memory["score"] += 1
        result = "✅ Correct!"
    else:
        result = f"❌ Incorrect. The correct answer was {correct}."
    accuracy = int((memory["score"] / memory["attempts"]) * 100)
    update_topic_schedule(topic, accuracy)
    save_memory(memory)
    return jsonify({ "result": result, "score": memory["score"], "attempts": memory["attempts"] })

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
    
    # Get onboarding data
    study_style = memory.get("study_style", "mixed")
    difficulty = memory.get("difficulty", "moderate")
    goals = memory.get("goals", "No goals set")
    learning_style = memory.get("learning_style", "No learning style specified")
    custom_topic = memory.get("custom_topic", "")
    focus_type = memory.get("focus_type", "general")
    chatbot_name = memory.get("chatbot_name", "Moxie")

    # Decide topic for this question
    topic = request.form.get("topic", custom_topic if custom_topic else "General Principles")

    messages = memory.get("messages", [])
    mimic_enabled = memory.get("use_comp_mimic", False)

    # Handle topic switching commands
    if user_input.startswith("/switch "):
        new_topic = user_input.replace("/switch ", "").strip()
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": f"🔄 Topic switched to **{new_topic}**."})
        memory["last_topic"] = new_topic
        save_memory(memory)
        return jsonify({"response": f"🔄 Topic switched to **{new_topic}**."})

    messages.append({"role": "user", "content": user_input})

    # Limit messages to last MAX_MESSAGES
    messages = messages[-MAX_MESSAGES:]

    # Retrieve similar questions for style mimic if enabled
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
        f"Today's topic: {topic}.\n"
        f"{examples}"
    )

    messages.insert(0, {"role": "system", "content": system_prompt})

    # Trim messages to fit within token limit
    while count_message_tokens(messages) > MAX_TOKENS and len(messages) > 1:
        messages.pop(1)

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        # Save updated memory
        memory["messages"] = messages
        memory["last_topic"] = topic
        save_memory(memory)

        return jsonify({"response": reply})

    except Exception as e:
        app.logger.error(f"OpenAI API error: {e}")
        return jsonify({"response": "Sorry, I am having trouble processing your request right now. Please try again later."}), 500

@app.route("/complete_setup", methods=["POST"])
def complete_setup():
    memory = load_memory()
    memory["setup_complete"] = True
    memory["comp_date"] = request.form.get("comp_date")
    memory["study_style"] = request.form.get("study_mode")
    memory["focus_type"] = request.form.get("topic_mode")
    memory["difficulty"] = request.form.get("difficulty")
    memory["goals"] = request.form.get("goals", "")
    memory["learning_style"] = request.form.get("learning_style", "")
    memory["custom_topic"] = request.form.get("custom_topic", "")
    chatbot_name = request.form.get("chatbot_name", "").strip()
    memory["chatbot_name"] = chatbot_name if chatbot_name else "Moxie"
    chatbot_avatar = request.form.get("chatbot_avatar", "").strip()
    if chatbot_avatar in ("girl", "boy"):
        memory["chatbot_avatar"] = "female" if chatbot_avatar == "girl" else "male"
    else:
        memory["chatbot_avatar"] = "robot"
    save_memory(memory)
    return redirect(url_for("home"))

@app.route("/quiz", methods=["POST"])
def quiz():
    memory = load_memory()
    mimic_enabled = memory.get("use_comp_mimic", False)
    prompt_base = "You're helping a medical student prepare for the COMP exam. Generate one COMP-style MCQ:"
    examples = ""
    if mimic_enabled:
        similar_qs = retrieve_similar_questions(prompt_base)
        examples = "\n\nUse this style as a guide:\n" + "\n\n".join(similar_qs)
    final_prompt = prompt_base + examples
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{ "role": "user", "content": final_prompt }]
    )
    return jsonify({ "question": response.choices[0].message.content })

@app.route("/quiz_page")
def quiz_page():
    return render_template("quiz.html")

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/dashboard/<topic>", methods=["GET"])
def topic_dashboard(topic):
    memory = load_memory()
    topic_data = memory.get("topics", {}).get(topic, {})
    return jsonify({
        "last_studied": topic_data.get("last_studied", "N/A"),
        "next_review": topic_data.get("next_review", "Not scheduled"),
        "accuracy": topic_data.get("accuracy", 0),
        "top_miss": topic_data.get("top_miss", "None yet"),
        "goals": topic_data.get("goals", "Not set"),
        "summary_codes": topic_data.get("summary_codes", []),
        "flashcards": topic_data.get("flashcards", [])
    })

@app.route("/calendar", methods=["GET"])
def review_calendar():
    memory = load_memory()
    calendar = {}
    for topic, data in memory.get("topics", {}).items():
        date = data.get("next_review")
        if date:
            calendar.setdefault(date, []).append(topic)
    today = datetime.today()
    future_14 = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(15)]
    result = {day: calendar.get(day, []) for day in future_14 if calendar.get(day)}
    return jsonify(result)

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