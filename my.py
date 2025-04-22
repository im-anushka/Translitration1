
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import difflib
import re
import os
import json
import requests
import zipfile
from googletrans import Translator

app = Flask(__name__)
CORS(app)  

# Device Configuration (CPU)
device = torch.device("cpu")

# Load Tokenizer & Model for English to Kannada
en_to_kn_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)
en_to_kn_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)
en_to_kn_model.load_state_dict(torch.load("engtokan_transliteration_model.pth", map_location=device))
en_to_kn_model.to(device)
en_to_kn_model.eval()

# Load Tokenizer & Model for Kannada to English
kn_to_en_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)
kn_to_en_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)
kn_to_en_model.load_state_dict(torch.load("kan_transliteration_model.pth", map_location=device))
kn_to_en_model.to(device)
kn_to_en_model.eval()

# Google Translator
translator = Translator()

# URL for the Kannada zip file on Hugging Face
KAN_ZIP_URL = "https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/main/kan.zip"
KAN_DATA_DIR = "kan_dataset"  # Directory where the dataset will be extracted

def download_and_extract_kan_dataset():
    if not os.path.exists(KAN_DATA_DIR):
        os.makedirs(KAN_DATA_DIR, exist_ok=True)
    zip_path = os.path.join(KAN_DATA_DIR, "kan.zip")
    if not os.path.exists(zip_path):
        print("Downloading kan.zip from Hugging Face...")
        response = requests.get(KAN_ZIP_URL)
        if response.status_code != 200:
            raise Exception("Failed to download kan.zip")
        with open(zip_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    marker_file = os.path.join(KAN_DATA_DIR, "extracted.txt")
    if not os.path.exists(marker_file):
        print("Extracting kan.zip...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(KAN_DATA_DIR)
        with open(marker_file, "w") as f:
            f.write("extracted")
        print("Extraction complete.")

try:
    download_and_extract_kan_dataset()
except Exception as e:
    print("Error downloading or extracting the dataset:", e)

import os
import json

def load_kannada_data():
    translit_dict = {}
    files = ["kan_train.json", "kan_valid.json", "kan_test.json"]
    for fname in files:
        file_path = os.path.join(KAN_DATA_DIR, fname)
        if os.path.exists(file_path):
            print(f"Loading data from {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                native = record.get("native word")
                                english = record.get("english word")
                                if native and english:
                                    translit_dict[native] = english  # Kannada to English
                                    translit_dict[english] = native  # English to Kannada
                            except Exception as e:
                                print(f"Error parsing line in {fname}: {e}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"File {file_path} not found.")
    print(f"Loaded {len(translit_dict)} transliteration pairs from Kannada data.")
    return translit_dict

# Load the data
kannada_translit = load_kannada_data()
kannada_translit["ಚಂದನ"] = "chandana"  # Manually add transliteration

def fallback_transliterate(name: str) -> str:
    mapping = {
       'ಅ': 'a', 'ಆ': 'aa', 'ಇ': 'i', 'ಈ': 'ee',
       'ಉ': 'u', 'ಊ': 'oo', 'ಋ': 'ru', 'ಎ': 'e',
       'ಏ': 'ee', 'ಐ': 'ai', 'ಒ': 'o', 'ಓ': 'oo',
       'ಔ': 'au', 'ಕ': 'ka', 'ಖ': 'kha', 'ಗ': 'ga',
       'ಘ': 'gha', 'ಙ': 'nga', 'ಚ': 'cha', 'ಛ': 'chha',
       'ಜ': 'ja', 'ಝ': 'jha', 'ಞ': 'nya'
    }
    return "".join(mapping.get(ch, ch) for ch in name)

def fallback_transliterate_en_to_kn(name: str) -> str:
    mapping = {
        'a': 'ಅ', 'aa': 'ಆ', 'i': 'ಇ', 'ee': 'ಈ',
        'u': 'ಉ', 'oo': 'ಊ', 'ru': 'ಋ', 'e': 'ಎ',
        'ai': 'ಐ', 'o': 'ಒ', 'oo': 'ಓ', 'au': 'ಔ',
        'ka': 'ಕ', 'kha': 'ಖ', 'ga': 'ಗ',
        'gha': 'ಘ', 'nga': 'ಙ', 'cha': 'ಚ', 'chha': 'ಛ',
        'ja': 'ಜ', 'jha': 'ಝ', 'nya': 'ಞ'
    }
    return "".join(mapping.get(ch, ch) for ch in name.split())


def remove_vowels(token: str) -> str:
    vowels = "aeiou"
    return "".join(ch for ch in token if ch not in vowels)

def kn_remove_vowels(token: str) -> str:
    vowels = "ಅಆಇಈಉಊಋಎಏಐಒಓಔ"  # Kannada vowels
    return "".join(ch for ch in token if ch not in vowels)


SALUTATIONS = {"mr", "mrs", "miss", "ms", "shri", "sri", "dr", "prof"}
def is_salutation(token: str) -> bool:
    token_clean = re.sub(r'[^\w]', '', token).lower()
    return token_clean in SALUTATIONS

KANNADA_SALUTATIONS = {"ಶ್ರೀ", "ಶ್ರಿ", "ಡಾ.", "ಡಾ", "ಪ್ರೊಫ.", "ಪ್ರೊಫ", "ಮಿಸಸ್", "ಮಿಸ", "ಮಿಸ್ಟರ್"}
def kn_is_salutation(token: str) -> bool:
    token_clean = re.sub(r'[^\w]', '', token).lower()  # Clean the token
    return token_clean in KANNADA_SALUTATIONS

def is_initial(token: str) -> bool:
    token_clean = token.replace('.', '')
    return len(token_clean) == 1 and token_clean.isalpha()

def kn_is_initial(token: str) -> bool:
    token_clean = token.replace('.', '')  # Remove any periods
    return len(token_clean) == 1 and token_clean.isalpha() 

def remove_trailing_a(token: str) -> str:
    if token.endswith("a") and len(token) > 1:
        return token[:-1]
    return token

def kn_remove_trailing_a(token: str) -> str:
    if token.endswith("ಅ") and len(token) > 1:  # Adjust for Kannada
        return token[:-1]
    return token

def phonetic_normalize(name: str) -> str:
    name = name.lower()
    name = re.sub(r'aa', 'a', name)
    name = re.sub(r'ee', 'i', name)
    name = re.sub(r'th', 't', name)
    return name.strip()

def kn_phonetic_normalize(name: str) -> str:
    name = name.lower()
    name = re.sub(r'ಅನ', 'ನ', name)  
    name = re.sub(r'ಆನ', 'ನ', name)  
    name = re.sub(r'ಇ', 'i', name)    
    name = re.sub(r'ಈ', 'i', name)  
    return name.strip()


def process_KtoE(name: str) -> tuple[list, list]:
    tokens = name.split()
    initials = []
    others = []
    for token in tokens:
        if kn_is_salutation(token):
            continue
        token_trans = kntoen(token)
        token_norm = kn_phonetic_normalize(token_trans)
        token_norm = kn_remove_vowels(token_norm)  
        if not kn_is_initial(token_norm):
            token_norm = kn_remove_trailing_a(token_norm)
            others.append(token_norm)
        else:
            initials.append(token_norm)
    return initials, others

def process_EtoK(name: str) -> tuple[list, list]:
    tokens = name.split()
    initials = []
    others = []
    for token in tokens:
        if is_salutation(token):
            continue
        token_trans = entokn(token)
        token_norm = phonetic_normalize(token_trans)
        token_norm = remove_vowels(token_norm)  
        if not is_initial(token_norm):
            token_norm = remove_trailing_a(token_norm)
            others.append(token_norm)
        else:
            initials.append(token_norm)
    return initials, others


class NameRequest:
    def __init__(self, name1: str, name2: str):
        self.name1 = name1
        self.name2 = name2

class NameMatchResponse:

    def __init__(self, match: bool, confidence: float, explanation: str):
        self.match = match
        self.confidence = confidence
        self.explanation = explanation

def is_kannada(text: str) -> bool:
    return any('\u0C80' <= char <= '\u0CFF' for char in text)

def is_english(text: str) -> bool:
    return all('A' <= char <= 'Z' or 'a' <= char <= 'z' or char.isspace() for char in text)


def translate_using_dataset(name: str) -> str:
    return kannada_translit.get(name, None)

def kntoen(name: str) -> str:
    if not is_english(name):  

        # Step 1: Check in preloaded dataset using translate_using_dataset
        transliterated_text = translate_using_dataset(name)
        if transliterated_text:
            return transliterated_text

        # Step 2: Use Kannada-to-English model
        try:
            inputs = kn_to_en_tokenizer(name, return_tensors="pt")
            with torch.no_grad():
                outputs = kn_to_en_model.generate(**inputs)
            transliterated_text = kn_to_en_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

            if transliterated_text:
                return transliterated_text
        except Exception as e:
            print(f"Model transliteration failed: {e}")

        # Fallback to simple transliteration for Kannada to English
        return fallback_transliterate(name)

    return name  


def entokn(name: str) -> str:
    if not is_kannada(name):

        # Step 1: Check in preloaded dataset using translate_using_dataset
        transliterated_text = translate_using_dataset(name)
        if transliterated_text:
            return transliterated_text

        # Step 2: Use English-to-Kannada model
        try:
            inputs = en_to_kn_tokenizer(name, return_tensors="pt")
            with torch.no_grad():
                outputs = en_to_kn_model.generate(**inputs)
            transliterated_text = en_to_kn_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

            if transliterated_text:
                return transliterated_text
        except Exception as e:
            print(f"Model transliteration failed: {e}")

        # Fallback to simple transliteration for English to Kannada
        return fallback_transliterate_en_to_kn(name)

    return name  


def compute_similarity(name1: str, name2: str) -> float:
    return difflib.SequenceMatcher(None, name1, name2).ratio()

def match_names_EC(name1: str, name2: str) -> tuple[bool, float, str]:
    original_name1 = name1
    original_name2 = name2

    initials1, tokens1 = process_KtoE(name1)
    initials2, tokens2 = process_KtoE(name2)

    explanation = f"Original names: '{original_name1}' and '{original_name2}'.\n"
    explanation += f"Extracted initials: {initials1} and {initials2}.\n"
    explanation += f"Remaining tokens: {tokens1} and {tokens2}.\n"

    if initials1 != initials2:
        explanation += "Initials do not match. Marking the entire name as mismatch."
        return False, 0.0, explanation

    joined1 = " ".join(tokens1)
    joined2 = " ".join(tokens2)
    similarity = compute_similarity(joined1, joined2)
    threshold = 0.95  # Primary threshold

    if similarity < 0.75:
        # Fallback to Google Translate
        try:
            translated_name1 = translator.translate(original_name1, src='kn', dest='en').text
            translated_name2 = translator.translate(original_name2, src='kn', dest='en').text
            fallback_similarity = compute_similarity(translated_name1, translated_name2)
            explanation += f"\nFallback similarity after Google Translate: {fallback_similarity:.2f}."
            return fallback_similarity >= threshold, fallback_similarity, explanation
        except Exception as e:
            explanation += f"\nGoogle Translate failed: {e}"

    is_match = similarity >= threshold
    explanation += f"Joined tokens: '{joined1}' and '{joined2}'.\n"
    explanation += f"Similarity of non-initial tokens: {similarity:.2f} (threshold: {threshold})."
    return is_match, similarity, explanation


def match_names_KC(name1: str, name2: str) -> tuple[bool, float, str]:
    original_name1 = name1
    original_name2 = name2

    initials1, tokens1 = process_EtoK(name1)
    initials2, tokens2 = process_EtoK(name2)

    explanation = f"Original names: '{original_name1}' and '{original_name2}'.\n"
    explanation += f"Extracted initials: {initials1} and {initials2}.\n"
    explanation += f"Remaining tokens: {tokens1} and {tokens2}.\n"

    if initials1 != initials2:
        explanation += "Initials do not match. Marking the entire name as mismatch."
        return False, 0.0, explanation

    joined1 = " ".join(tokens1)
    joined2 = " ".join(tokens2)
    similarity = compute_similarity(joined1, joined2)
    threshold = 0.95  # Primary threshold

    if similarity < 0.75:
        # Fallback to Google Translate
        try:
            translated_name1 = translator.translate(original_name1, src='en', dest='kn').text
            translated_name2 = translator.translate(original_name2, src='en', dest='kn').text
            fallback_similarity = compute_similarity(translated_name1, translated_name2)
            explanation += f"\nFallback similarity after Google Translate: {fallback_similarity:.2f}."
            return fallback_similarity >= threshold, fallback_similarity, explanation
        except Exception as e:
            explanation += f"\nGoogle Translate failed: {e}"

    is_match = similarity >= threshold
    explanation += f"Similarity of non-initial tokens: {similarity:.2f} (threshold: {threshold})."
    return is_match, similarity, explanation

@app.route("/")
def home():
    return "Flask Transliteration API is Running!"


@app.route("/transliterate/en-to-kn", methods=["POST"])
def transliterate_en_to_kn():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Text is required"}), 400

    transliterated_text = entokn(text)
    return jsonify({"input": text, "transliteration": transliterated_text})

@app.route("/transliterate/kn-to-en", methods=["POST"])
def transliterate_kn_to_en():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Text is required"}), 400

    transliterated_text = kntoen(text)
    return jsonify({"input": text, "transliteration": transliterated_text})


@app.route("/match-names", methods=["POST"])
def match_names():
    data = request.get_json()
    name_request = NameRequest(name1=data.get("name1", ""), name2=data.get("name2", ""))
    
    if not name_request.name1 or not name_request.name2:
        return jsonify({"error": "Both name1 and name2 are required"}), 400

    match_flag, confidence, explanation = match_names_EC(name_request.name1, name_request.name2)
    response = NameMatchResponse(match=match_flag, confidence=confidence, explanation=explanation)
    return jsonify(response.__dict__)

@app.route("/match-names2", methods=["POST"])
def match_names2():
    data = request.get_json()
    name_request = NameRequest(name1=data.get("name1", ""), name2=data.get("name2", ""))
    
    if not name_request.name1 or not name_request.name2:
        return jsonify({"error": "Both name1 and name2 are required"}), 400

    match_flag, confidence, explanation = match_names_KC(name_request.name1, name_request.name2)
    response = NameMatchResponse(match=match_flag, confidence=confidence, explanation=explanation)
    return jsonify(response.__dict__)

if __name__ == "__main__":
    print("Script is running!")
    app.run(host="0.0.0.0", port=5000, debug=True)