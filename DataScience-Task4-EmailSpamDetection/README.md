# Task 4: Email / SMS Spam Detection with Machine Learning

## Project Overview
This project implements a Natural Language Processing (NLP) binary classification pipeline that accurately classifies text messages as either legitimate (Ham) or Spam.

## Dataset
- **Source**: UCI Machine Learning Repository (SMS Spam Collection)
- **Total Records**: 5,572 messages
- **Distribution**: 86.6% Ham, 13.4% Spam

## Methodology
1. **Preprocessing**: Normalization, case folding, and non-alphanumeric character removal.
2. **Feature Extraction**: TF-IDF vectorizer removing standard English stop words.
3. **Data Splitting**: Stratified 80/20 train/test split.
4. **Classification Algorithms**:
   - Multinomial Naive Bayes
   - Logistic Regression

## Model Evaluation
| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) |
|---|---|---|---|---|
| **Multinomial Naive Bayes** | ~98.0% | ~0.99 | ~0.87 | ~0.93 |
| **Logistic Regression** | ~97.5% | ~0.98 | ~0.84 | ~0.90 |

> **Note on Metric Selection**: Precision is prioritized for spam filtering to minimize false positives, preventing legitimate emails from being incorrectly marked as spam.
