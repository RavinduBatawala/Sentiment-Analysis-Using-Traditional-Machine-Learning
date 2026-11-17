# Sentiment Analysis Using Traditional Machine Learning

**PUSL 3205 – Large Language Models and Generative AI | Lab 1**

Binary sentiment classification (positive / negative) of short reviews using text preprocessing, Bag-of-Words / TF-IDF features and classical machine learning models.

## Dataset
[UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences): 3,000 sentences from three sources, each perfectly balanced.

| Source | Sentences | Positive | Negative |
|---|---|---|---|
| Amazon (product reviews) | 1,000 | 500 | 500 |
| IMDb (movie reviews) | 1,000 | 500 | 500 |
| Yelp (restaurant reviews) | 1,000 | 500 | 500 |

## Workflow
1. **Load** the three files and combine them into one DataFrame (`quoting=csv.QUOTE_NONE` is needed, otherwise 252 IMDb rows are silently lost)
2. **Preprocess**: lowercase, expand negations (`don't` → `do not`), remove punctuation, normalise whitespace
3. **Explore**: class balance, sentence lengths, most frequent words per class
4. **Split**: 80/20 stratified train/test split
5. **Vectorise**: Bag-of-Words and TF-IDF with unigrams + bigrams
6. **Train**: Logistic Regression, Linear SVM and Multinomial Naive Bayes
7. **Evaluate**: Accuracy, Precision, Recall, F1 and a confusion matrix
8. **Test** on new, unseen sentences

## Results

| Features | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **TF-IDF** | **Linear SVM** | **0.845** | **0.827** | **0.873** | **0.849** |
| BoW | Logistic Regression | 0.840 | 0.825 | 0.863 | 0.844 |
| BoW | Linear SVM | 0.835 | 0.821 | 0.857 | 0.839 |
| TF-IDF | Logistic Regression | 0.830 | 0.800 | 0.880 | 0.838 |
| TF-IDF | Naive Bayes | 0.825 | 0.814 | 0.843 | 0.828 |
| BoW | Naive Bayes | 0.822 | 0.810 | 0.840 | 0.825 |

## Key Findings
- Keeping negation words and using **bigrams** lets the model learn phrases like *"not good"*, which became one of its strongest negative features.
- *"not"* appears about 4× more often in negative reviews than in positive ones.
- Bag-of-words models fail on **double negation**, **mixed sentiment** and **sarcasm** (e.g. *"Oh great, it broke after one day"* is predicted positive), which motivates context-aware models such as transformers and LLMs.

## Project Structure
```
├── data/                     # UCI dataset (3 labelled .txt files)
├── notebooks/
│   └── lab1_sentiment.ipynb  # full analysis
├── requirements.txt
└── README.md
```

## How to Run
```bash
git clone https://github.com/RavinduBatawala/Sentiment-Analysis-Using-Traditional-Machine-Learning.git
cd Sentiment-Analysis-Using-Traditional-Machine-Learning
python -m venv venv
venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
Then open `notebooks/lab1_sentiment.ipynb` and run all cells.

## Tech Stack
Python · Pandas · NumPy · NLTK · Scikit-learn · Matplotlib · Seaborn