# DebateBias-8K

**Surfacing Subtle Stereotypes: A Multilingual, Debate-Oriented Evaluation of Modern LLMs**  
Accepted at **LREC 2026** | Palma, Mallorca

## Overview

DebateBias-8K is a multilingual, debate-oriented benchmark introduced to study how safety-aligned large language models may reproduce harmful stereotypes in open-ended generation settings.

The project framework is centered around:
- **4 bias domains**: Backwardness, Religion, Terrorism, Women's Rights
- **7 languages**: English, Chinese, Arabic, Hindi, Korean, Swahili, Nigerian Pidgin
- **5 demographic categories**: Western, Arabs, South Asians, Indians, Africans

## Repository Status

This repository currently provides the **codebase and documentation** associated with the DebateBias-8K pipeline.

## Repository Structure

```text
DebateBias-8K/
├── README.md
├── requirements.txt
├── src/
│   ├── seed_generation.py        # Phase 1: Semi-automatic seed creation
│   ├── translation.py            # Phase 2: Multilingual translation
│   ├── back_translation.py       # Translation quality audit
│   ├── subtopic_discovery.py     # Subtopic clustering and labeling
│   ├── probing.py                # Target model probing
│   └── classification.py         # Response classification
├── configs/
│   └── .env.example              # API key template
└── docs/
    └── usage.md                  # Optional: setup / pipeline notes
```

## Pipeline

The DebateBias-8K pipeline consists of five main stages:

| Stage | Script                  | Model(s)                                        | Description                                                   |
| ----- | ----------------------- | ----------------------------------------------- | ------------------------------------------------------------- |
| S1    | `seed_generation.py`    | GPT-4o                                          | Generate seed prompts and expand them via in-context learning |
| S2    | `subtopic_discovery.py` | text-embedding-3-small, GPT-4o                  | Cluster prompts and label subtopics                           |
| S3    | `translation.py`        | GPT-4o                                          | Translate English prompts into the target languages           |
| S3b   | `back_translation.py`   | GPT-4o, `paraphrase-multilingual-mpnet-base-v2` | Audit translation quality through back-translation            |
| S4    | `probing.py`            | GPT-4o, Claude, DeepSeek, LLaMA                 | Probe target models with the benchmark framework              |
| S5    | `classification.py`     | GPT-4o, DeepSeek, LLaMA-3-8B                    | Classify generated responses using a majority-vote setup      |

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/muhammed-saeed/DebateBias-8K.git
cd DebateBias-8K
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp configs/.env.example .env
# Edit .env with the required API keys
```

## Models

### Target Models

* GPT-4o
* Claude 3.5 Haiku
* DeepSeek-Chat
* LLaMA-3-70B

### Pipeline Models

* GPT-4o for seed generation, prompt expansion, translation, back-translation, and classification
* DeepSeek-Chat for classification
* LLaMA-3-8B for classification

### Evaluation Encoders

* `text-embedding-3-small` for subtopic discovery
* `paraphrase-multilingual-mpnet-base-v2` for prompt similarity and back-translation auditing



## Citation

```bibtex
@inproceedings{saeed-etal-2026-surfacing,
    title     = {Surfacing Subtle Stereotypes: A Multilingual, Debate-Oriented Evaluation of Modern {LLMs}},
    author    = {Saeed, Muhammed Yahia Gaffar and Abdul-Mageed, Muhammad and Shehata, Shady},
    booktitle = {Proceedings of the 15th Language Resources and Evaluation Conference (LREC 2026)},
    year      = {2026},
    address   = {Palma, Mallorca}
}
```

## License

This repository is released for **research and academic use**. Please refer to the conference paper for discussion of ethical considerations and limitations.

## Disclaimer

This project studies harmful stereotypes in language model generation. Some scripts and documentation may reference sensitive demographic categories and socially harmful associations strictly for research, evaluation, and analysis purposes.