# SMS Spam Detection using NLP  
**Building a Text Classification Pipeline & Word Embedding Exploration**

---

## Project Overview

This project implements an **end-to-end Natural Language Processing (NLP) pipeline** for **SMS spam detection**, framed around a real-world telecom use case.

**Stakeholder:** Telecom operators / SMS gateway providers  
**Problem:** Automatically detect and filter spam SMS messages before delivery  
**Objective:** Build and compare multiple text representations and classification models to understand trade-offs between:
- Generative vs. discriminative classifiers
- Sparse vs. dense text representations
- Performance, interpretability, and computational efficiency

The project follows a complete NLP workflow:
1. Text preprocessing and normalization
2. Feature engineering using sparse and dense representations
3. Model training using classical machine-learning algorithms
4. Evaluation and comparative analysis

---

## Dataset Description

**Dataset:** SMS Spam Collection Dataset (UCI Machine Learning Repository)

- Total messages: ~5,500
- Classes:
  - `ham` (legitimate messages)
  - `spam` (unsolicited / fraudulent messages)
- Task: Binary text classification

The dataset contains short, noisy SMS messages with URLs, numbers, and informal language, making it well-suited for evaluating preprocessing and feature-engineering choices in a telecom context.

**Source:**  
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

---

## Results Summary

The following approaches were implemented and systematically compared to evaluate their effectiveness for SMS spam detection.

### Feature Representations
- **Bag-of-Words (BoW)**
- **TF-IDF** (unigrams + bigrams )
- **Word2Vec embeddings** (document-level vectors obtained by averaging word embeddings)

### Models
- **Multinomial Naïve Bayes**  
  *(Generative classifier trained on sparse feature representations)*
- **Logistic Regression**  
  *(Discriminative classifier trained on both sparse and dense representations)*

### Key Observations
- Multinomial Naïve Bayes performs strongly on Bag-of-Words features due to the presence of clear spam-indicator keywords commonly found in unsolicited messages.
- Logistic Regression combined with TF-IDF achieves competitive or superior performance by assigning higher importance to informative words and short phrases (bigrams) frequently associated with spam.
- Word2Vec-based document embeddings generally underperform sparse representations for SMS data, as averaging embeddings weakens strong keyword signals in very short texts.
- Sparse representations are computationally efficient, highly interpretable, and better suited for large-scale SMS filtering systems deployed in telecom environments.

### Evaluation Metrics
Models were evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

Special emphasis was placed on **precision**, as falsely classifying legitimate messages as spam (false positives) has a higher real-world cost.

Detailed numerical results and full classification reports are available in the accompanying Jupyter notebook.