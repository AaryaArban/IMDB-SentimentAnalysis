# Comparative Sentiment Analysis of IMDB Movie Reviews

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-brightgreen.svg)
![Pandas](https://img.shields.io/badge/pandas-purple.svg)

## Project Overview

This project is a comprehensive study for the **AIDS 2 Lab**, conducting sentiment analysis on the IMDB Movie Review dataset. The primary goal is to compare the performance, efficiency, and practical trade-offs of classical machine learning techniques versus various deep learning architectures for a binary text classification task.

We build and evaluate four distinct models:
1.  **Simple Deep Learning:** A simple 1D Convolutional Neural Network (CNN)
2.  **Regularized Deep Learning:** An improved 1D CNN with Dropout (to address overfitting)
3.  **Sequential Deep Learning:** A Bidirectional Long Short-Term Memory (LSTM) network

The analysis provides a clear visual and statistical comparison of each model's accuracy, F1-score, and training time, culminating in a final recommendation.

## Dataset

This project uses the **IMDB Dataset of 50K Movie Reviews**, a standard benchmark for binary sentiment classification.

* **Source:** [IMDB Dataset of 50K Movie Reviews on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* **Structure:** The dataset contains 50,000 reviews, each labeled as "positive" or "negative".
* **Sampling:** For practical training times, this project uses a random sample of **20,000 reviews** from the full dataset.

## Project Workflow

The notebook `IMDB_SentimentAnalysis.ipynb` is structured as a complete machine learning pipeline:

1.  **Data Loading:** The `IMDB Dataset.csv` is loaded into a pandas DataFrame and sampled down to 20,000 reviews.
2.  **Preprocessing:**
    * **Text Cleaning:** A function `clean_review_with_stopwords_and_stemming` is defined and applied to all reviews. This process involves:
        * Unescaping HTML (e.g., `&quot;` -> `"`)
        * Removing all HTML tags (e.g., `<br />`)
        * Converting text to lowercase
        * Removing all punctuation and non-alphabetic characters
        * Removing common English **stopwords** (e.g., "the", "a", "is")
        * Applying **Porter Stemming** to reduce words to their root (e.g., "movies" -> "movi")
    * **Label Encoding:** The categorical labels ("positive", "negative") are converted to numerical format (`1`, `0`).
3.  **Data Splitting:** The 20,000 samples are split into an 80% training set (16,000) and a 20% test set (4,000) using a `random_state` for reproducibility.
4.  **Vectorization:** The cleaned text data is converted into two distinct numerical formats for the different models:
    * **TF-IDF Vectors:** A matrix of TF-IDF features is created for the Scikit-learn baseline (`TfidfVectorizer`).
    * **Padded Sequences:** The text is tokenized into integer sequences, a vocabulary is built (top 10,000 words), and all sequences are padded/truncated to a uniform length of 200 (`Tokenizer`, `pad_sequences`).
5.  **Model Training & Evaluation:** Each of the four models is trained on the training data and evaluated on the test data.
6.  **Comparative Analysis:** The results (Accuracy, F1-Score, Training Time) from all models are collected into a final table and visualized with bar charts to identify the best model.
7.  **Prediction & Saving:** A prediction pipeline is demonstrated on new, unseen reviews, and the best-performing models are saved to disk.

---

## Models Explored

A key goal of this project was to demonstrate a thorough understanding of different modeling approaches and their challenges, particularly overfitting.

### 1. Original CNN (The Overfitting Problem)
* **Purpose:** To demonstrate a common pitfall in deep learning: **overfitting**.
* **Technique:** A simple 1D Convolutional Neural Network (CNN) is built. CNNs are effective for text as they act as pattern detectors, identifying key n-grams (phrases) that are strong indicators of sentiment.
* **Analysis:** This model was intentionally trained without regularization. The resulting accuracy/loss plot clearly shows the training accuracy reaching ~100% while validation accuracy stalls, and validation loss begins to *increase*—a classic sign of the model memorizing the training data instead of learning to generalize.

### 2. Improved CNN (with Dropout)
* **Purpose:** To demonstrate the *solution* to overfitting using regularization.
* **Technique:** This model uses the *same architecture* as the original CNN but adds a **`Dropout(0.5)`** layer. Dropout randomly deactivates 50% of the neurons during each training step, forcing the model to learn more robust and redundant features.
* **Analysis:** The history plot for this model shows the training and validation lines tracking each other closely. This proves that dropout was highly effective in mitigating overfitting, leading to a model that generalizes better to unseen data.

### 3. Bidirectional LSTM
* **Purpose:** To explore a more advanced architecture specifically designed for sequential data.
* **Technique:** A **Long Short-Term Memory (LSTM)** network reads the review word-by-word, maintaining a "memory" that captures context and long-range dependencies (e.g., how the word "not" modifies a word 10 places later).
* **`Bidirectional` Wrapper:** The LSTM is wrapped in a `Bidirectional` layer, meaning it processes the text *twice*: once from left-to-right and once from right-to-left. This gives the model a complete contextual understanding of every word in the sequence.

---

## Results & Conclusion

The performance of all four models was compiled and compared.

### Summary of Results

| Model | F1-Score | Train Time (s) | Accuracy |
|:---|---:|---:|---:|
| Original CNN | 0.8557 | 185.30 | 0.8555 |
| Improved CNN | 0.8516 | 235.78 | 0.8520 |
| Bidirectional LSTM | 0.8556 | 594.61 | 0.8570 |

### Conclusion

The **`Tfidf + Logistic Regression`** model was the clear winner and the recommended solution for this task.

* **Performance:** It achieved the highest F1-Score (0.8867) and Accuracy (0.8842).
* **Efficiency:** It was **exponentially faster** to train (2.7 seconds) compared to the deep learning models (235-594 seconds).

This project highlights a crucial concept in machine learning: **model complexity does not always equal better performance**. While the Bidirectional LSTM is a more advanced architecture, its high complexity led to the longest training time without providing any significant performance benefit over the much simpler, faster, and more interpretable classical model. The simple CNN, while fast, was prone to overfitting, which required regularization (Dropout) to fix.

---

## How to Install and Run

1.  Clone this repository:
    ```sh
    git clone https://github.com/AaryaArban/IMDB-SentimentAnalysis
    cd IMDB-SentimentAnalysis
    ```

2.  Install the required dependencies:
    ```sh
    pip install tensorflow pandas numpy scikit-learn nltk seaborn matplotlib
    ```

3.  Run the Jupyter notebook:
    ```sh
    jupyter notebook IMDB_SentimentAnalysis.ipynb
    ```
    *Note: The first time you run the notebook, it will download the `stopwords` package from NLTK.*

