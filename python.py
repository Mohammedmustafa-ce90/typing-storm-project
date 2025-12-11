import os
import time
import random
import csv
import json
import numpy as np
from difflib import SequenceMatcher
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

# ================= FILE NAMES =================
TEXT_FILE = "english_texts.txt"
SESSIONS_FILE = "typing_sessions.csv"
XP_FILE = "player_xp.json"


# ================= SKILL MODEL =================

X_train = np.array([
    [10, 60], [18, 75], [25, 82],
    [32, 86], [40, 89], [52, 92],
    [65, 95], [80, 97]
])
y_train = np.array([
    "Beginner", "Beginner", "Intermediate",
    "Intermediate", "Intermediate", "Advanced",
    "Advanced", "Advanced"
])

skill_model = KNeighborsClassifier(n_neighbors=3)
skill_model.fit(X_train, y_train)

# KNN training accuracy
SKILL_TRAIN_ACC = skill_model.score(X_train, y_train)

# The words that will be used to generate the text target
ADJECTIVES = ["quick", "bright", "strange", "tiny", "massive", "silent", "complex", "confident"]
NOUNS = ["fox", "student", "keyboard", "algorithm", "device", "system", "model", "neuron"]
VERBS = ["jumps", "calculates", "types", "observes", "crashes", "learns", "predicts", "analyzes"]
ADVERBS = ["quickly", "silently", "awkwardly", "smoothly", "unexpectedly", "carefully"]


# ================ ERROR LETTERS =================

def extract_error_letters(target, user_input):
    """
    Compare target text and user input positionally.
    Whenever the user misses or mistypes a character,
    record the correct character from the target (if it's a letter).
    """
    errors = []
    max_len = max(len(target), len(user_input))
    for i in range(max_len):
        t_char = target[i] if i < len(target) else ""
        u_char = user_input[i] if i < len(user_input) else ""
        if t_char != u_char and t_char.isalpha():
            errors.append(t_char.lower())
    return "".join(errors)


# ================ TEXT GENERATION =================

def generate_sentence():
    return f"The {random.choice(ADJECTIVES)} {random.choice(NOUNS)} {random.choice(VERBS)} {random.choice(ADVERBS)}."


def auto_generate_text_file(path=TEXT_FILE, count=300):
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(count):
            f.write(generate_sentence() + "\n")
    print(f"[AUTO] Generated {count} sentences into {path}")


if not os.path.exists(TEXT_FILE):
    auto_generate_text_file(TEXT_FILE)


def load_corpus():
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


CORPUS = load_corpus()


# ================= LEVEL SYSTEM =================

# To open the file that stores XP
def load_xp():
    if not os.path.exists(XP_FILE):
        return 0
    try:
        with open(XP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return int(data.get("xp", 0))
    except Exception:
        return 0


# To save the XP back to file
def save_xp(xp):
    with open(XP_FILE, "w", encoding="utf-8") as f:
        json.dump({"xp": int(xp)}, f)


# To get level from XP
def get_level(xp):
    return xp // 100


# ============== SESSION HISTORY =================

def load_history():
    if not os.path.exists(SESSIONS_FILE):
        return []
    hist = []
    with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                try:
                    hist.append((float(row[0]), float(row[1])))
                except ValueError:
                    continue
    return hist


def save_session(wpm, acc, target, user_input):
    # get error letters for this session
    error_letters = extract_error_letters(target, user_input)

    with open(SESSIONS_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # store wpm, accuracy, and the string of error letters
        writer.writerow([wpm, acc, error_letters])


def get_weak_letters(max_sessions=5, top_k=3):
    """
    Look at the last `max_sessions` rows in the CSV,
    read the error_letters column, and find the most common letters.
    """
    if not os.path.exists(SESSIONS_FILE):
        return []

    with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return []

    # Take last N sessions (from the end)
    recent_rows = rows[-max_sessions:]

    freq = {}
    for row in recent_rows:
        if len(row) >= 3:
            error_letters = row[2]
            for ch in error_letters:
                if ch.isalpha():
                    freq[ch] = freq.get(ch, 0) + 1

    if not freq:
        return []

    # Sort by frequency (highest first) and take top_k letters
    sorted_letters = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [letter for letter, count in sorted_letters[:top_k]]


def train_predictors(history):  # (52,40)
    if len(history) < 2:
        return None, None
    X = []
    y_wpm = []
    y_acc = []
    for i in range(len(history) - 1):
        X.append([history[i][0], history[i][1]])
        y_wpm.append(history[i + 1][0])
        y_acc.append(history[i + 1][1])
    X = np.array(X)
    y_wpm = np.array(y_wpm)
    y_acc = np.array(y_acc)
    reg_wpm = LinearRegression().fit(X, y_wpm)
    reg_acc = LinearRegression().fit(X, y_acc)
    return reg_wpm, reg_acc


# ============== AI METRICS =================

def show_ai_metrics(history):
    """
    Display basic evaluation metrics for the AI models:
    - KNN skill classifier training accuracy
    - Linear Regression R² scores for next WPM and next Accuracy (if enough history)
    """
    # KNN classifier accuracy on its small training set
    
    print(f"AI skill model (KNN) training accuracy: {SKILL_TRAIN_ACC * 100:.2f}%")

    # Need at least 2 sessions to evaluate regression models
    if len(history) < 2:
        print("AI prediction models (Linear Regression): not enough data yet (need at least 2 sessions).")
        return

    # Build data for regression evaluation from history
    X = []
    y_wpm = []
    y_acc = []
    for i in range(len(history) - 1):
        X.append([history[i][0], history[i][1]])   # previous session (wpm, acc)
        y_wpm.append(history[i + 1][0])            # next session wpm
        y_acc.append(history[i + 1][1])            # next session accuracy

    X = np.array(X)
    y_wpm = np.array(y_wpm)
    y_acc = np.array(y_acc)

    reg_wpm_eval = LinearRegression().fit(X, y_wpm)
    reg_acc_eval = LinearRegression().fit(X, y_acc)

    r2_wpm = reg_wpm_eval.score(X, y_wpm)
    r2_acc = reg_acc_eval.score(X, y_acc)

    print(f"AI next-WPM model (Linear Regression) R²: {r2_wpm:.2f}")
    print(f"AI next-Accuracy model (Linear Regression) R²: {r2_acc:.2f}")


# =============== MISSION SYSTEM =================

MISSIONS = {
    "Easy": {
        "sentences": 1,
        "desc": "Warm-up: short sentence, focus on accuracy.",
        "multiplier": 1.0,
    },
    "Medium": {
        "sentences": 2,
        "desc": "Steady run: medium length, balance speed and accuracy.",
        "multiplier": 1.5,
    },
    "Hard": {
        "sentences": 3,
        "desc": "Challenge: longer text, push your speed.",
        "multiplier": 2.0,
    }
}


def estimate_skill_from_history(history):
    if not history:
        return "Beginner"
    last_wpm, last_acc = history[-1]
    skill = skill_model.predict(np.array([[last_wpm, last_acc]]))[0]
    return skill


def choose_mission(history):
    skill = estimate_skill_from_history(history)
    if skill == "Beginner":
        difficulty = "Easy"
    elif skill == "Intermediate":
        difficulty = "Medium"
    else:
        difficulty = "Hard"
    mission = MISSIONS[difficulty].copy()
    mission["difficulty"] = difficulty
    mission["skill_estimate"] = skill
    return mission


def build_target_text(sentences_count, focus_letters=None):
    """
    Build mission text using `sentences_count` sentences.
    If focus_letters is given, prefer sentences that contain those letters.
    """
    if not focus_letters:
        # no specific weak letters -> random sentences
        return " ".join(random.choice(CORPUS) for _ in range(sentences_count))

    focus_letters = [c.lower() for c in focus_letters]
    chosen = []
    attempts = 0

    # Try to pick sentences that include at least one weak letter
    while len(chosen) < sentences_count and attempts < 1000:
        s = random.choice(CORPUS)
        s_lower = s.lower()
        if any(letter in s_lower for letter in focus_letters):
            chosen.append(s)
        attempts += 1

    # If we didn't find enough, fill the rest with random sentences
    while len(chosen) < sentences_count:
        chosen.append(random.choice(CORPUS))

    return " ".join(chosen)


# ================= TYPING TEST =================

def terminal_test():
    # Load previous data
    history = load_history()
    reg_wpm, reg_acc = train_predictors(history)
    total_xp = load_xp()
    level = get_level(total_xp)

    # Show AI model metrics
    show_ai_metrics(history)

    # AI-chosen mission
    mission = choose_mission(history)

    # Find weak letters from last 5 sessions
    weak_letters = get_weak_letters(max_sessions=5, top_k=3)

    if weak_letters:
        print(f"\nAI focus letters (your weak spots): {', '.join(weak_letters)}")
    else:
        print("\nAI focus letters: none yet, general practice.")

    # Build mission text, biased towards weak letters
    target = build_target_text(mission["sentences"], focus_letters=weak_letters)

    print("\n===================================")
    print("          TYPING QUEST")
    print("===================================\n")
    print(f"Current XP: {total_xp} | Level: {level}")
    print(f"AI thinks your skill is around: {mission['skill_estimate']}")
    print(f"Assigned Mission: [{mission['difficulty']}] {mission['desc']}")
    print(f"Text length: {mission['sentences']} sentence(s)")
    print("\nYour mission text:\n")
    print("> " + target + "\n")

    input("Press ENTER to start the mission...")
    start = time.time()
    user_input = input("\nStart typing below:\n> ")
    elapsed = time.time() - start

    # ----------- WPM -----------
    chars = len(user_input)
    words = chars / 5
    minutes = elapsed / 60 if elapsed > 0 else 1  # essential
    wpm = words / minutes

    # ----------- Accuracy (similarity-based) -----------
    accuracy = SequenceMatcher(None, user_input, target).ratio() * 100

    # ----------- Skill prediction (KNN) -----------
    features = np.array([[wpm, accuracy]])
    skill_now = skill_model.predict(features)[0]

    # ----------- Save session -----------
    save_session(wpm, accuracy, target, user_input)

    # ----------- XP gain -----------
    mult = mission["multiplier"]
    session_xp = max(0, int(wpm * (accuracy / 100.0) * mult))
    total_xp += session_xp
    save_xp(total_xp)
    new_level = get_level(total_xp)

    # ----------- Predict next test performance -----------
    if reg_wpm is not None and reg_acc is not None:
        pred_wpm_raw = reg_wpm.predict(features)[0]
        pred_acc_raw = reg_acc.predict(features)[0]

        # Global sane clamp
        pred_wpm_raw = max(0.0, min(200.0, pred_wpm_raw))
        pred_acc_raw = max(0.0, min(100.0, pred_acc_raw))

        # Anchor around current performance
        lower_wpm = 0.8 * wpm
        upper_wpm = 1.3 * wpm
        pred_wpm = max(lower_wpm, min(upper_wpm, pred_wpm_raw))

        lower_acc = 0.8 * accuracy
        upper_acc = min(100.0, 1.05 * accuracy)
        pred_acc = max(lower_acc, min(upper_acc, pred_acc_raw))
    else:
        pred_wpm = None
        pred_acc = None

    # ---------------------- RESULTS ----------------------
    print("\n===================================")
    print("              RESULTS")
    print("===================================\n")
    print(f"Time: {elapsed:.2f} s")
    print(f"WPM: {wpm:.2f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Skill (this run): {skill_now}")
    print(f"Mission difficulty: {mission['difficulty']}")
    print(f"XP gained: {session_xp}")
    print(f"Total XP: {total_xp} | Level: {new_level}")

    if pred_wpm is not None:
        print("\n--- AI Prediction for Next Test ---")
        print(f"Predicted next WPM: {pred_wpm:.2f}")
        print(f"Predicted next Accuracy: {pred_acc:.2f}%")
    else:
        print("\nNot enough past data for next-test prediction yet.")

    print("\nMission complete. Nice work.\n")


if __name__ == "__main__":
    while True:
        terminal_test()
        again = input("Run another mission? (y/n): ").strip().lower()
        if again != "y":
            break