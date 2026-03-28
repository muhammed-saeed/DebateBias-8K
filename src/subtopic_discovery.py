"""
Subtopic discovery via k-means clustering over OpenAI embeddings.
Clusters 1,200 English prompts into 10 subtopics per domain,
then uses GPT-4 to generate descriptive labels for each cluster.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from tqdm import tqdm
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
LABELING_MODEL = "gpt-4o"
LABELING_TEMPERATURE = 0.3
N_CLUSTERS = 10

INPUT_CSV = "data/prompts/en/generated_bias_prompts.csv"
OUTPUT_DIR = "data/subtopic_analysis"

sns.set(style="whitegrid", font_scale=1.3)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def get_embedding(text):
    try:
        response = openai.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def label_cluster(texts):
    prompt = (
        "You will be given a list of prompts about a common theme.\n"
        "Please return a very short, descriptive 3 to 5 word subtopic name "
        "that captures the shared theme.\n\n"
        "Now summarize this group:\n"
    )
    prompt += "\n".join(f"- {t}" for t in texts[:5])
    prompt += "\n\nShort Subtopic Name:"

    try:
        response = openai.chat.completions.create(
            model=LABELING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=LABELING_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Labeling error: {e}")
        return "Unnamed Subtopic"


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df["combined"] = df["title"].astype(str) + " " + df["prompt"].astype(str)

    duplicates = df.duplicated(subset=["title", "prompt"], keep=False).sum()
    print(f"Found {duplicates} duplicated title+prompt pairs.")
    df = df.drop_duplicates(subset=["title", "prompt"]).reset_index(drop=True)

    print("Generating embeddings...")
    tqdm.pandas()
    df["embedding"] = df["combined"].progress_apply(get_embedding)
    df = df[df["embedding"].notnull()].reset_index(drop=True)

    for category in df["category"].unique():
        print(f"\nProcessing: {category}")
        df_cat = df[df["category"] == category].copy()

        if len(df_cat) < N_CLUSTERS:
            print(f"  Skipping {category}: too few prompts.")
            continue

        embeddings = np.vstack(df_cat["embedding"].values)
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
        df_cat["subtopic"] = kmeans.fit_predict(embeddings)

        subtopic_names = {}
        for sub in sorted(df_cat["subtopic"].unique()):
            examples = df_cat[df_cat["subtopic"] == sub]["combined"].tolist()
            subtopic_names[sub] = label_cluster(examples)

        df_cat["subtopic_name"] = df_cat["subtopic"].map(subtopic_names)

        counts = df_cat["subtopic_name"].value_counts()
        plt.figure(figsize=(10, 6))
        bars = sns.barplot(x=counts.index, y=counts.values, palette="muted")
        plt.title(f"{category} - Subtopic Distribution", fontsize=16, weight="bold")
        plt.ylabel("Number of Prompts")
        plt.xlabel("Subtopics")
        plt.xticks(rotation=45, ha="right")

        for bar, count in zip(bars.patches, counts.values):
            bars.annotate(
                f"{count}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=11,
            )

        plt.tight_layout()
        safe_name = category.replace(" ", "_").lower()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"{safe_name}_subtopics.pdf"),
            format="pdf", bbox_inches="tight",
        )
        plt.close()

        csv_path = os.path.join(OUTPUT_DIR, f"{safe_name}_subtopics.csv")
        df_cat[["title", "subtopic", "subtopic_name"]].to_csv(csv_path, index=False)

    print(f"\nAll plots and CSVs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
