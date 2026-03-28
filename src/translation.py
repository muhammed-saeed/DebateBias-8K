"""
Phase 3: Multilingual translation.
Translates 1,200 English prompts into 6 target languages using GPT-4 (temperature=0.3)
with schema validation and retry logic.
"""

import os
import json
import csv
import time
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o"
TEMPERATURE = 0.3
MAX_RETRIES = 3

TARGET_LANGUAGES = {
    "ar": "Arabic",
    "hi": "Hindi",
    "sw": "Swahili",
    "pcm": "Nigerian Pidgin",
    "zh": "Chinese",
    "ko": "Korean",
}

INPUT_PATH = "data/prompts/en/generated_bias_prompts.json"
OUTPUT_DIR = "data/prompts"


def is_valid_prompt(text):
    if text.count("{Demographic List}") != 1:
        return False
    if text.count("{Demographic") != 1:
        return False
    number_matches = re.findall(r"(?m)^([1-3])\.\s", text)
    counts = {str(i): number_matches.count(str(i)) for i in range(1, 4)}
    return all(v == 1 for v in counts.values())


def is_valid_title(translated_title, original_title):
    if "{Demographic List}" in translated_title:
        return False
    original_has_number = bool(re.search(r"\d", original_title))
    translated_has_number = bool(re.search(r"\d", translated_title))
    return original_has_number == translated_has_number


def translate_text(content, lang_name, retry_attempt=0):
    system_prompt = (
        f"You are a professional translator. Translate the following text into {lang_name}. "
        "Preserve technical phrasing, placeholders like {Demographic List}, and the original formatting. "
        "Do not translate placeholders or variables that resemble markup (e.g., {variable_name}). "
        "After you translate the points under the focus areas make sure that you keep the numbers "
        "1, 2, 3 in the translated text."
    )

    if retry_attempt > 0:
        system_prompt += (
            f"\n\nIMPORTANT: This is retry attempt {retry_attempt}. Previous translation failed validation. "
            "Please ensure your translation:"
            "\n1. Contains exactly ONE occurrence of '{Demographic List}'"
            "\n2. Preserves numbered points (1., 2., 3.) on separate lines"
            "\n3. Does not add or remove any numbered points"
            "\n4. Does not add {Demographic List} to the title if translating a title"
        )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return None


def translate_with_retries(content, lang_name, validator, original=None):
    result = translate_text(content, lang_name)
    retry = 0

    if original is not None:
        valid = result and is_valid_title(result, original)
    else:
        valid = result and is_valid_prompt(result)

    while not valid and retry < MAX_RETRIES:
        retry += 1
        print(f"  Retrying translation (attempt {retry}/{MAX_RETRIES})")
        result = translate_text(content, lang_name, retry)
        time.sleep(1.0)

        if original is not None:
            valid = result and is_valid_title(result, original)
        else:
            valid = result and is_valid_prompt(result)

    return result if valid else None


def run():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        original_prompts = json.load(f)

    for lang_code, lang_name in TARGET_LANGUAGES.items():
        print(f"\nTranslating into {lang_name}...")

        lang_folder = os.path.join(OUTPUT_DIR, lang_code)
        os.makedirs(lang_folder, exist_ok=True)

        log_path = os.path.join(lang_folder, "translated_log.json")
        translated_log = {}
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                translated_log = json.load(f)

        translated_data = {}
        csv_rows = []
        count = 0

        for category, prompts in original_prompts.items():
            if category not in translated_data:
                translated_data[category] = []

            for p in prompts:
                title_key = f"{category}::{p['title']}"
                if title_key in translated_log:
                    continue

                translated_title = translate_with_retries(
                    p["title"], lang_name, is_valid_title, original=p["title"]
                )
                translated_prompt = translate_with_retries(
                    p["prompt"], lang_name, is_valid_prompt
                )

                if not translated_title or not translated_prompt:
                    invalid_path = os.path.join(lang_folder, "invalid_prompts.json")
                    invalid = []
                    if os.path.exists(invalid_path):
                        with open(invalid_path, "r", encoding="utf-8") as f:
                            invalid = json.load(f)
                    invalid.append({"category": category, "title": p["title"]})
                    with open(invalid_path, "w", encoding="utf-8") as f:
                        json.dump(invalid, f, indent=2, ensure_ascii=False)
                    print(f"  Invalid after retries, logged: {p['title']}")
                    continue

                translated_data[category].append(
                    {"title": translated_title, "prompt": translated_prompt}
                )
                csv_rows.append(
                    {"category": category, "title": translated_title, "prompt": translated_prompt}
                )
                translated_log[title_key] = {
                    "category": category,
                    "title": translated_title,
                    "prompt": translated_prompt,
                }

                count += 1
                print(f"  [{count}] {p['title']}")

                if count % 10 == 0:
                    _save_progress(lang_folder, translated_data, csv_rows, translated_log, log_path)
                    csv_rows = []

                time.sleep(1.0)

        _save_progress(lang_folder, translated_data, csv_rows, translated_log, log_path)
        print(f"Done for {lang_name}: {count} prompts translated.")


def _save_progress(lang_folder, translated_data, csv_rows, translated_log, log_path):
    json_path = os.path.join(lang_folder, "translated_prompts.json")
    existing = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    for cat, prompts in translated_data.items():
        if cat not in existing:
            existing[cat] = []
        existing_titles = {e["title"] for e in existing[cat]}
        for prompt in prompts:
            if prompt["title"] not in existing_titles:
                existing[cat].append(prompt)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(lang_folder, "translated_prompts.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["category", "title", "prompt"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(csv_rows)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(translated_log, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run()
