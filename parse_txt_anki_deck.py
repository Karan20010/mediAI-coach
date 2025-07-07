import os
import json

# --- Config ---
INPUT_FILE = "user_data/anki_deck/Comp Anki Deck.txt"
OUTPUT_FILE = "user_data/anki_flashcards/COMP_Anki_Deck.json"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# --- Parser ---
def parse_txt_to_json(path, deduplicate=True):
    cards = []
    seen = set()
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                question, answer = parts[0].strip(), parts[1].strip()
                key = (question, answer)
                if not deduplicate or key not in seen:
                    cards.append({"question": question, "answer": answer})
                    seen.add(key)
            elif line.strip():  # non-empty but malformed
                print(f"⚠️ Skipped malformed line {line_num}: {line.strip()}")
    
    return cards

# --- Main Execution ---
try:
    cards = parse_txt_to_json(INPUT_FILE)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(cards, out, indent=2)
    print(f"✅ Saved {len(cards)} cards to {OUTPUT_FILE}")
except FileNotFoundError:
    print(f"❌ File not found: {INPUT_FILE}")
except Exception as e:
    print(f"❌ Error: {e}")