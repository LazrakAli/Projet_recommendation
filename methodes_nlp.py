from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
from gensim.models import LdaModel

def baseline(comments_tuples, top_n=10):
        """
        Return the most frequent words in the comments.
        
        :param comments_tuples: List of tuples in the form (rank, neighbor_id, comment)
        :param top_n: Number of top frequent words to return
        :return: List of most frequent words
        """
        all_words = []
        
        for _, _, comment in comments_tuples:
            all_words.extend(comment)
        
        word_counts = Counter(all_words)
        most_common_words = word_counts.most_common(top_n)
        
        return [word for word, count in most_common_words]
    

def tf_idf(comments_tuples, top_n=10):
    """
    Extract keywords from multiple preprocessed comments using TF-IDF.
    
    :param comments_tuples: List of tuples in the form (rank, neighbor_id, comment)
    :param top_n: Number of top keywords to return
    :return: List of top keywords for the combined comments
    """
    # Extract comments from tuples
    comments = [' '.join(words) for _, _, words in comments_tuples]
    
    # Compute TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(comments)
    
    # Extract Keywords
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    tfidf_scores = list(zip(feature_names, tfidf_scores))
    tfidf_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top_n keywords
    top_keywords = [word for word, score in tfidf_scores[:top_n]]
    
    return top_keywords

def lda(comments_tuples, num_topics=5, num_keywords=10):
    # Extract preprocessed comments from tuples
    processed_comments = [words for _, _, words in comments_tuples]
    
    # Create a dictionary and corpus
    dictionary = corpora.Dictionary(processed_comments)
    corpus = [dictionary.doc2bow(comment) for comment in processed_comments]
    
    # Train the LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    
    # Extract keywords
    keywords = {}
    for i in range(num_topics):
        # Extract and format topic keywords
        topic_keywords = lda_model.show_topic(i, topn=num_keywords)
        keywords[f'Topic {i}'] = [word for word, _ in topic_keywords]
    
    return keywords
