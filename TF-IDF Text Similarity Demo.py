"""
TF-IDF Text Similarity Demo
---------------------------
This script demonstrates how to:
1. Represent text data as numerical vectors using TF-IDF.
2. Compare a query sentence against a set of documents.
3. Find the most similar sentences using cosine similarity.

It's designed as an educational project for beginners in NLP & ML.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------------
# 1. Example corpus (dataset)
# -------------------------
# Normally you would load data from a file, but here we hardcode a few example sentences.
corpus = [
    "I love watching movies on weekends",
    "Do you have a movie recommendation?",
    "Any good film suggestions for tonight?",
    "Let's grab some coffee tomorrow",
    "Artificial intelligence is transforming the world",
    "Machine learning and AI are closely related fields",
    "I need to finish my homework today",
    "This restaurant serves great pizza",
]

# -------------------------
# 2. Build TF-IDF vectorizer
# -------------------------
# TfidfVectorizer converts text into numerical vectors.
# - max_features=500 limits vocabulary size to 500 unique terms.
# - stop_words='english' removes common words like 'the', 'is', 'and'.
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')

# Fit the vectorizer to the corpus and transform into vectors (document-term matrix).
# Each row = document, each column = word feature.
X = vectorizer.fit_transform(corpus)

print("Vocabulary (features):")
print(vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix shape:", X.shape)  # (n_documents, n_features)

# -------------------------
# 3. Query sentence
# -------------------------
query = "Can you suggest me a movie to watch?"

# Transform the query into the same vector space as the corpus.
q_vec = vectorizer.transform([query])  # shape (1, n_features)

# -------------------------
# 4. Compute similarity
# -------------------------
# Cosine similarity measures the angle between vectors, giving us a similarity score.
# - 1.0 means exactly the same direction (identical),
# - 0 means no similarity.
scores = cosine_similarity(q_vec, X).flatten()

# -------------------------
# 5. Find top-k most similar sentences
# -------------------------
topk = 3  # number of results to show
topk_idx = scores.argsort()[::-1][:topk]  # sort scores descending

print(f"\nQuery: {query}\n")
print("Top similar sentences in the corpus:")
for idx in topk_idx:
    print(f"- {corpus[idx]} (similarity: {scores[idx]:.3f})")

# -------------------------
# Notes / Educational Takeaways
# -------------------------
# - TF (Term Frequency): how often a word appears in a document.
# - IDF (Inverse Document Frequency): downweights common words across many documents.
# - TF-IDF = TF * IDF → gives higher weight to rare-but-important words.
#
# - Cosine Similarity: compares two vectors regardless of magnitude, focusing on their orientation.
#   This is useful because document lengths differ (short vs long text).
#
# - Limitations: TF-IDF does not understand synonyms or context. For example,
#   "film" and "movie" are treated as different words. More advanced methods like
#   word embeddings (Word2Vec, BERT, Sentence Transformers) capture semantic meaning.
