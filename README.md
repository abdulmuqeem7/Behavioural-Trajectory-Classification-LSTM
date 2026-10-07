# evs-lstm-depression-trajectory


An MSc dissertation project that moves beyond static "depressed vs. not depressed" classification by modelling how a user's mental health **changes over time** — classifying behavioral trajectories as **Escalating**, **Recovering**, or **Stable**.

## Problem

Most depression-detection NLP models classify a single post or user snapshot in isolation. This ignores an important clinical signal: **direction of change**. This project asks a different question — given a sequence of a user's posts, is their mental health getting worse, improving, or staying stable?

## Approach

Three models were built and compared on the same dataset:

| Model | Description | Accuracy |
|---|---|---|
| Baseline 1 | TF-IDF + Logistic Regression | 47.38% |
| Baseline 2 | MentalBERT (transformer, fine-tuned) | 47.77% |
| **EVS-LSTM** (proposed) | Engineered behavioral features (EVS score, sentiment, keyword signals) fed as sequences through an LSTM | **58.56%** |

The two baselines analyse posts in isolation and land at near-identical performance, even with a domain-pretrained transformer. **EVS-LSTM** explicitly models temporal/behavioral progression across a user's post history, which produced a meaningful accuracy improvement, particularly for the harder "Escalating" and "Recovering" classes.

### EVS-LSTM classification report

<img width="578" height="212" alt="image" src="https://github.com/user-attachments/assets/d4c99bd7-2275-4b88-9575-db71b1409e51" />


Model interpretability was also explored using SHAP to understand which engineered features drove predictions.

## Data

This project uses the **CLEF eRisk** dataset (Reddit posts labelled for depression-related research). The dataset is **not included in this repository** — it is subject to CLEF eRisk's own data usage agreement and is available to researchers directly from the organizers. No raw post text, usernames, or dataset files are published here.

## Tech Stack

Python · pandas · scikit-learn · TensorFlow/Keras · PyTorch · Hugging Face Transformers · NLTK · SHAP

## Author

Feel free to connect if you'd like to discuss the methodology or results — always happy to chat about NLP, applied ML, or responsible AI in sensitive domains.
