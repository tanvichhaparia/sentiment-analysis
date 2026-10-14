"""
Unsupervised Fashion Topic Discovery
======================================
Uses TF-IDF + NMF (Non-Negative Matrix Factorization) and LDA (Latent Dirichlet
Allocation) to automatically discover fashion topics from raw social media text,
WITHOUT relying on predefined keyword dictionaries.

Why NMF over LDA for short texts?
- NMF with TF-IDF produces more interpretable, coherent topics on short documents
  like social media captions and comments (Shi et al., 2018).
- LDA is included as an alternative for comparison.

References:
- Lee & Seung (1999). "Learning the parts of objects by non-negative matrix factorization."
- Blei, Ng & Jordan (2003). "Latent Dirichlet Allocation."
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


def discover_topics_nmf(texts, n_topics=10, n_top_words=8, max_features=3000):
    """
    Discover topics using TF-IDF + Non-Negative Matrix Factorization.
    
    Parameters
    ----------
    texts : list of str
        Preprocessed (cleaned) text documents.
    n_topics : int
        Number of topics to extract.
    n_top_words : int
        Number of top words per topic to return.
    max_features : int
        Maximum vocabulary size for TF-IDF.
        
    Returns
    -------
    dict with keys:
        'topics'        : list of dicts, each with 'id', 'top_words', 'label'
        'doc_topic_matrix' : numpy array (n_docs x n_topics), topic weights per document
        'feature_names' : list of str, the vocabulary
    """
    # Build TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        max_df=0.90,       # Ignore terms appearing in >90% of docs (too common)
        min_df=3,          # Ignore terms appearing in <3 docs (too rare)
        ngram_range=(1, 2) # Unigrams + bigrams to capture phrases like "cargo pants"
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Fit NMF
    nmf_model = NMF(
        n_components=n_topics,
        random_state=42,
        max_iter=300,
        init='nndsvda'    # Deterministic initialisation for reproducibility
    )
    doc_topic_matrix = nmf_model.fit_transform(tfidf_matrix)

    # Extract top words per topic
    topics = []
    for topic_idx, topic_vec in enumerate(nmf_model.components_):
        top_word_indices = topic_vec.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_word_indices]
        topics.append({
            'id': topic_idx,
            'top_words': top_words,
            'label': f"Topic {topic_idx + 1}: {', '.join(top_words[:3])}"
        })

    return {
        'topics': topics,
        'doc_topic_matrix': doc_topic_matrix,
        'feature_names': feature_names.tolist()
    }


def discover_topics_lda(texts, n_topics=10, n_top_words=8, max_features=3000):
    """
    Discover topics using Count Vectorizer + Latent Dirichlet Allocation.
    Provided as a comparison to NMF.
    """
    count_vectorizer = CountVectorizer(
        max_features=max_features,
        max_df=0.90,
        min_df=3,
        ngram_range=(1, 2)
    )
    count_matrix = count_vectorizer.fit_transform(texts)
    feature_names = count_vectorizer.get_feature_names_out()

    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method='online'
    )
    doc_topic_matrix = lda_model.fit_transform(count_matrix)

    topics = []
    for topic_idx, topic_vec in enumerate(lda_model.components_):
        top_word_indices = topic_vec.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_word_indices]
        topics.append({
            'id': topic_idx,
            'top_words': top_words,
            'label': f"Topic {topic_idx + 1}: {', '.join(top_words[:3])}"
        })

    return {
        'topics': topics,
        'doc_topic_matrix': doc_topic_matrix,
        'feature_names': feature_names.tolist()
    }


def assign_dominant_topic(doc_topic_matrix, topic_labels):
    """
    For each document, assign the topic with the highest weight.
    
    Returns
    -------
    list of str : dominant topic label per document
    """
    dominant_indices = np.argmax(doc_topic_matrix, axis=1)
    return [topic_labels[i] for i in dominant_indices]


def run_topic_discovery(df, text_column='clean_text', n_topics=10, method='nmf'):
    """
    Main entry point: runs topic discovery on a DataFrame and returns results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain `text_column` with preprocessed text.
    text_column : str
        Column name containing cleaned text.
    n_topics : int
        Number of topics to discover.
    method : str
        'nmf' or 'lda'.
        
    Returns
    -------
    results : dict
        Contains 'topics', 'dominant_topics' (list), 'doc_topic_matrix'.
    """
    texts = df[text_column].fillna('').tolist()

    # Filter out empty strings
    valid_mask = [len(t.strip()) > 0 for t in texts]
    valid_texts = [t for t, m in zip(texts, valid_mask) if m]

    if len(valid_texts) < n_topics * 2:
        print(f"WARNING: Only {len(valid_texts)} valid texts; reducing n_topics to {max(2, len(valid_texts) // 2)}")
        n_topics = max(2, len(valid_texts) // 2)

    print(f"Running {method.upper()} topic discovery with {n_topics} topics on {len(valid_texts)} documents...")

    if method == 'lda':
        result = discover_topics_lda(valid_texts, n_topics=n_topics)
    else:
        result = discover_topics_nmf(valid_texts, n_topics=n_topics)

    topic_labels = [t['label'] for t in result['topics']]

    # Assign dominant topic to each valid document
    dominant_topics_valid = assign_dominant_topic(result['doc_topic_matrix'], topic_labels)

    # Map back to full DataFrame (empty docs get "Unknown")
    dominant_topics_full = []
    valid_idx = 0
    for m in valid_mask:
        if m:
            dominant_topics_full.append(dominant_topics_valid[valid_idx])
            valid_idx += 1
        else:
            dominant_topics_full.append("Unknown")

    result['dominant_topics'] = dominant_topics_full
    print(f"Topic discovery complete. Discovered {n_topics} topics.")

    return result


def save_topics(result, output_path='reports/discovered_topics.json'):
    """Save discovered topics to JSON (without the large doc_topic_matrix)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_data = {
        'topics': result['topics'],
        'topic_distribution': {}
    }
    # Count documents per topic
    from collections import Counter
    dist = Counter(result['dominant_topics'])
    save_data['topic_distribution'] = dict(dist)

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=4)
    print(f"Discovered topics saved to {output_path}")


if __name__ == "__main__":
    # Quick test
    test_texts = [
        "oversized denim jacket streetwear look",
        "quiet luxury minimalist outfit",
        "y2k aesthetic coquette ribbon bows",
        "sustainable fashion thrift vintage",
        "cargo pants athleisure sneakers",
    ] * 20  # Repeat to have enough data

    result = discover_topics_nmf(test_texts, n_topics=3)
    for t in result['topics']:
        print(t['label'])
