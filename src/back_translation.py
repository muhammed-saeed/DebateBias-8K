"""
Back-translation quality audit.
Samples prompts per language-domain cell, back-translates via GPT-4,
and computes cosine similarity using paraphrase-multilingual-mpnet-base-v2.
"""

import os
import json
import csv
import time
import re
import random
import numpy as np
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TRANSLATION_MODEL = "gpt-4o"
TRANSLATION_TEMPERATURE = 0.3
ENCODER_MODEL = "paraphrase-multilingual-mpnet-base-v2"
SAMPLES_PER_CATEGORY = 50
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
OUTPUT_DIR = "data/back_translation_audit"

print(f"Loading encoder: {ENCODER_MODEL}")
encoder = SentenceTransformer(ENCODER_MODEL)
print("Encoder loaded.")


def translate_text(content, from_lang, to_lang, retry_attempt=0):
    system_prompt = (
        f"You are a professional translator. Translate the following text from {from_lang} to {to_lang}. "
        "Preserve technical phrasing, placeholders like {Demographic List}, and the original formatting. "
        "Do not translate placeholders or variables that resemble markup (e.g., {variable_name}). "
        "After you translate the points under the focus areas make sure that you keep the numbers "
        "1, 2, 3 in the translated text."
    )

    if retry_attempt > 0:
        system_prompt += (
            f"\n\nIMPORTANT: This is retry attempt {retry_attempt}. "
            "Ensure the translation contains exactly ONE '{Demographic List}' "
            "and preserves numbered points (1., 2., 3.) on separate lines."
        )

    try:
        response = client.chat.completions.create(
            model=TRANSLATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=TRANSLATION_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        time.sleep(5)
        return None


def compute_similarity(text1, text2):
    cleaned1 = re.sub(r"\{[^}]*\}", "", text1)
    cleaned2 = re.sub(r"\{[^}]*\}", "", text2)
    emb1 = encoder.encode(cleaned1, convert_to_tensor=True)
    emb2 = encoder.encode(cleaned2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        original_prompts = json.load(f)

    results = {}

    for lang_code, lang_name in TARGET_LANGUAGES.items():
        print(f"\nProcessing {lang_name}...")

        lang_folder = os.path.join(OUTPUT_DIR, lang_code)
        os.makedirs(lang_folder, exist_ok=True)

        results[lang_code] = {"language": lang_name, "categories": {}, "all_similarities": []}

        for category, prompts in original_prompts.items():
            print(f"  Category: {category}")
            sample = random.sample(prompts, min(SAMPLES_PER_CATEGORY, len(prompts)))
            category_similarities = []

            for p in tqdm(sample, desc=f"  {category}", unit="prompt"):
                translated = translate_text(p["prompt"], "English", lang_name)
                if not translated:
                    continue

                back_translated = translate_text(translated, lang_name, "English")
                if not back_translated:
                    continue

                sim = compute_similarity(p["prompt"], back_translated)
                category_similarities.append(sim)
                results[lang_code]["all_similarities"].append(sim)
                time.sleep(1.0)

            if category_similarities:
                results[lang_code]["categories"][category] = {
                    "n": len(category_similarities),
                    "mean": float(np.mean(category_similarities)),
                    "median": float(np.median(category_similarities)),
                    "std": float(np.std(category_similarities)),
                }

            detail_path = os.path.join(lang_folder, f"{category}_similarity.csv")
            with open(detail_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["similarity"])
                for s in category_similarities:
                    writer.writerow([s])

        all_sims = results[lang_code]["all_similarities"]
        if all_sims:
            results[lang_code]["overall"] = {
                "n": len(all_sims),
                "mean": float(np.mean(all_sims)),
                "median": float(np.median(all_sims)),
                "std": float(np.std(all_sims)),
                "iqr": float(np.percentile(all_sims, 75) - np.percentile(all_sims, 25)),
            }

        with open(os.path.join(lang_folder, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results[lang_code], f, indent=2, default=str)

        print(f"  Done: median={results[lang_code].get('overall', {}).get('median', 'N/A')}")

    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "code", "n", "mean", "median", "std", "iqr"])
        for lang_code, data in results.items():
            o = data.get("overall", {})
            writer.writerow([
                data["language"], lang_code,
                o.get("n", 0), o.get("mean", 0), o.get("median", 0),
                o.get("std", 0), o.get("iqr", 0),
            ])

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    run()
