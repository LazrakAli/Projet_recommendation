from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
from gensim.models import LdaModel
from summa import keywords
import yake

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

def textrank(comments_tuples, ratio=0.2):
    """
    Extract keywords from a list of comments using TextRank algorithm via the summa library.

    :param comments: List of strings, where each string is a comment.
    :param ratio: Float, controls the fraction of text to output as keywords.
    :return: A string containing the extracted keywords.
    """
    # Join all comments into a single text
    comments = [' '.join(words) for _, _, words in comments_tuples]
    text = ' '.join(comments)
    
    return keywords.keywords(text, ratio=ratio)

def yake_extractor(comments_tuples, num_keywords=10, deduplication_threshold=0.9, n_gram_size=3):
    """
    Extract keywords from a list of tokenized comments using the YAKE algorithm.

    :param comments_tuples: List of lists, where each inner list is a list of tokens from a comment.
    :param num_keywords: Number of keywords to extract.
    :param deduplication_threshold: Threshold to use for deduplication; the lower, the more aggressive.
    :param n_gram_size: The maximum length of multi-word keywords (n-grams).
    :return: Dictionary of comments and their corresponding list of keywords.
    """
    # Initialize YAKE keyword extractor
    language = "fr"  # Assuming the language is English
    max_ngram_size = n_gram_size
    deduplication_thresold = deduplication_threshold
    num_of_keywords = num_keywords
    processed_comments = [words for _, _, words in comments_tuples]
    
    extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, top=num_of_keywords, features=None)

    # Process each tokenized comment
    keywords_per_comment = {}
    for index, tokens in enumerate(processed_comments):
        # Join tokens into a single string
        comment_text = ' '.join(tokens)
        # Extract keywords
        keywords = extractor.extract_keywords(comment_text)
        # Store keywords, optionally could use index or any other identifier for each comment
        keywords_per_comment[index] = [kw[0] for kw in keywords]

    return keywords_per_comment