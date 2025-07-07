import json
import os

# --- File paths ---
INPUT_FILE = "user_data/anki_flashcards/COMP_Anki_Deck_Merged_With_Salvaged.json"
OUTPUT_FILE = "user_data/anki_flashcards/COMP_Anki_Deck_Cleaned_2.json"
INVALID_FILE = "user_data/anki_flashcards/Invalid_Cards_For_Review.json"
STILL_SKIPPED_FILE = "user_data/anki_flashcards/Still_Skipped_Invalids.json"

# --- Ensure folder exists ---
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# --- Validate & clean ---
def validate_and_clean_cards(path):
    with open(path, "r", encoding="utf-8") as f:
        cards = json.load(f)

    seen = set()
    cleaned = []
    skipped = []
    duplicates = 0

    for idx, card in enumerate(cards):
        q = card.get("question", "").strip()
        a = card.get("answer", "").strip()

        if not q or not a:
            skipped.append({ "index": idx, "question": q, "answer": a })
            continue

        key = (q, a)
        if key in seen:
            duplicates += 1
            continue

        seen.add(key)
        cleaned.append({ "question": q, "answer": a })

    return cleaned, skipped, duplicates

# --- Run process ---
if __name__ == "__main__":
    cleaned_cards, skipped_cards, duplicate_count = validate_and_clean_cards(INPUT_FILE)

    # --- Save cleaned deck ---
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(cleaned_cards, out, indent=2, ensure_ascii=False)

    # --- Save invalid cards ---
    with open(INVALID_FILE, "w", encoding="utf-8") as out:
        json.dump(skipped_cards, out, indent=2, ensure_ascii=False)

    # --- Compare skipped against cleaned to find which are still missing ---
    cleaned_set = set((c["question"].strip(), c["answer"].strip()) for c in cleaned_cards)
    still_skipped = [card for card in skipped_cards
                     if (card["question"].strip(), card["answer"].strip()) not in cleaned_set]

    with open(STILL_SKIPPED_FILE, "w", encoding="utf-8") as out:
        json.dump(still_skipped, out, indent=2, ensure_ascii=False)

    # --- Summary ---
    print(f"✅ Cleaned deck saved to {OUTPUT_FILE}")
    print(f"📦 Cards kept: {len(cleaned_cards)}")
    print(f"♻️ Duplicates removed: {duplicate_count}")
    print(f"❌ Skipped invalid cards: {len(skipped_cards)}")
    print(f"📁 Still missing in cleaned output: {len(still_skipped)} (saved to {STILL_SKIPPED_FILE})")