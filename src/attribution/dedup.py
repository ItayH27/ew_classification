"""
Near-duplicate article filtering for the attribution stage.

World News API aggregates coverage from many outlets, and several of those outlets
frequently republish the same underlying wire-service story (e.g. AP/Reuters syndication)
under a different headline and URL, with the body text largely unchanged. Exact URL/title
matching misses this, and without any dedup, `assess_attribution`'s per-article Gemini
loop would double/triple-count one underlying story, biasing its score toward whatever
verdict Gemini gives that one story. This compares article text (not identifiers) via
TF-IDF cosine similarity to collapse those near-duplicates before classification.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_DEFAULT_THRESHOLD = 0.8


def deduplicate_articles(
    articles: list[dict],
    threshold: float = _DEFAULT_THRESHOLD,
) -> tuple[list[dict], int]:
    """
    Collapses near-duplicate articles (same underlying story, different outlet) via TF-IDF
    cosine similarity over each article's text (falling back to title if text is missing).

    Clusters articles via union-find over the `similarity >= threshold` graph — not a
    greedy "compare only to kept representatives" scan — so transitive near-duplicate
    chains (A~B~C where sim(A, C) alone might fall just under threshold) are still merged.
    Each cluster's representative is its first-appearing article (World News API's own
    relevance-sorted order), since union-find always attaches the larger-index root to the
    smaller-index root.

    Args:
        articles: article dicts as returned by search_articles() (each with at least
            "text" and/or "title").
        threshold: cosine similarity at/above which two articles are treated as the same
            underlying story.

    Returns:
        (deduped_articles, num_duplicates_removed) — deduped_articles is one representative
        per cluster, in original relative order; num_duplicates_removed = len(articles) -
        len(deduped_articles).
    """
    if len(articles) <= 1:
        return list(articles), 0

    texts = [(article.get("text") or article.get("title") or "").strip() for article in articles]

    try:
        vectors = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts)
    except ValueError:
        # Every article's fallback text was empty/stopwords-only — no signal to compare,
        # safest default is "nothing is a duplicate".
        print("deduplicate_articles: empty vocabulary across all articles, skipping dedup")
        return list(articles), 0

    similarity = cosine_similarity(vectors)

    n = len(articles)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[max(root_i, root_j)] = min(root_i, root_j)

    kept_indices = sorted({find(i) for i in range(n)})
    deduped = [articles[i] for i in kept_indices]
    return deduped, n - len(deduped)
