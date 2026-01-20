# 🚗 Analyzing Car Reviews with LLMs

**Car-ing is Sharing — LLM Prototype Project**

## 📌 Overview

This project is a **proof-of-concept chatbot backend** built for *Car-ing is Sharing*, a car sales and rental company.
It demonstrates how **Large Language Models (LLMs)** can be leveraged to handle diverse customer-facing NLP tasks such as:

* Sentiment analysis and evaluation
* Machine translation and quality assessment
* Extractive question answering
* Text summarization

The pipeline processes a small dataset of car reviews and produces structured outputs and evaluation metrics that can later be integrated into a chatbot or customer insights platform.

---

## 🎯 Objectives

The prototype fulfills the following tasks:

### 1️⃣ Sentiment Classification

* Use a **pre-trained sentiment analysis LLM** to classify the sentiment of **five car reviews** from `car_reviews.csv`
* Store raw model outputs in `predicted_labels`
* Convert predictions into binary labels `{0,1}` stored in `predictions`
* Evaluate performance using:

  * **Accuracy** → `accuracy_result`
  * **F1 Score** → `f1_result`

---

### 2️⃣ English → Spanish Translation + Evaluation

* Extract the **first two sentences** of the **first review**
* Translate the text using an **English-to-Spanish translation LLM**
* Store the translated output in `translated_review`
* Evaluate translation quality using **BLEU score**

  * References provided in `reference_translations.txt`
  * Store metric in `bleu_score`

---

### 3️⃣ Extractive Question Answering

* Focus on the **second review**, which highlights brand aspects
* Use an extractive QA model:

  ```
  deepset/minilm-uncased-squad2
  ```
* Ask the question:

  ```
  "What did he like about the brand?"
  ```
* Use:

  * `question` → the query
  * `context` → the review text
* Store the extracted answer in `answer`

---

### 4️⃣ Review Summarization

* Summarize the **last review** in the dataset
* Target length: **~50–55 tokens**
* Store output in `summarized_text`

---

## 🧠 Models Used

| Task               | Model Type                        |
| ------------------ | --------------------------------- |
| Sentiment Analysis | Pre-trained sentiment classifier  |
| Translation        | English → Spanish translation LLM |
| Question Answering | `deepset/minilm-uncased-squad2`   |
| Summarization      | Pre-trained summarization LLM     |

All models are loaded via **Hugging Face Transformers**.

---

## 📁 Project Structure

```
Analyzing-Car-Reviews-with-LLMs/
├── data/
│   ├── car_reviews.csv                # Input dataset (5 car reviews)
│   └── reference_translations.txt     # Reference translations for BLEU
│
├── main.ipynb                             # Runs all tasks end-to-end
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Full Pipeline

```bash
open main.ipynb
```

This will:

* Load the dataset
* Execute all LLM tasks
* Compute evaluation metrics

---

## 📊 Outputs & Variables

| Variable            | Description                            |
| ------------------- | -------------------------------------- |
| `predicted_labels`  | Raw sentiment predictions from the LLM |
| `predictions`       | Binary sentiment labels `{0,1}`        |
| `accuracy_result`   | Sentiment classification accuracy      |
| `f1_result`         | Sentiment classification F1 score      |
| `translated_review` | Spanish translation of review text     |
| `bleu_score`        | BLEU score for translation quality     |
| `question`          | QA input question                      |
| `context`           | QA context (review text)               |
| `answer`            | Extracted QA answer                    |
| `summarized_text`   | ~50–55 token review summary            |

---

## 🛠️ Dependencies

Key libraries used:

* `transformers`
* `torch`
* `pandas`
* `scikit-learn`
* `nltk`
* `evaluate`

---

## 🚀 Future Extensions

* Wrap pipelines into a **FastAPI chatbot backend**
* Add **multi-language sentiment analysis**
* Store results in a vector database for retrieval
* Integrate conversational memory (RAG)

---

## 📄 License

MIT License

---
