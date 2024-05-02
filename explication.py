import pandas as pd                        # Importe la bibliothèque pandas et la renomme en pd
import spacy                               # Importe la bibliothèque spacy
import numpy as np                         # Importe la bibliothèque numpy et la renomme en np
import surprise
import os
from surprise import Dataset, Reader, KNNWithZScore  # Importe les classes Dataset, Reader et KNNWithZScore depuis la bibliothèque surprise
from collections import defaultdict       # Importe la classe defaultdict depuis la bibliothèque collections
import openai

data = pd.read_csv("BDD/avis_sans_outliers.csv")

data_cleaned = data.drop(columns=['Unnamed: 0', 'url', 'title_review', 'date_published'])
data_cleaned['comment'] = data_cleaned['comment'].fillna('')  # Fill missing comments

# Prepare data for surprise
reader = Reader(rating_scale=(1, 10))  # Assuming rating scale is from 1 to 10
data_surprise = Dataset.load_from_df(data_cleaned[['author', 'title', 'note']], reader)

# Build full trainset
trainset = data_surprise.build_full_trainset()

# Initialize KNN with Z-Score algorithm for user-based collaborative filtering
algo = KNNWithZScore(sim_options={'name': 'cosine', 'user_based': True}, k=20, min_k=1)
algo.fit(trainset)



def filter_comments_by_word_count(comments, min_word_count):
    """
    Filters a list of comments to include only those with a word count greater or equal to min_word_count.
    
    Args:
    comments (list of str): A list of comments.
    min_word_count (int): The minimum number of words required for a comment to be included in the return list.
    
    Returns:
    list of str: A list containing only comments that meet or exceed the word count requirement.
    """
    # Filter comments based on word count
    filtered_comments = [comment for comment in comments if len(comment.split()) >= min_word_count]
    return filtered_comments


def get_top_comments(user_id, game_title, N, word_count_threshold):
    """
    Retrieves the top N comments from the nearest neighbors of a specified user about a specific game,
    using the KNNWithZScore algorithm from Surprise. Only comments with a word count above a certain threshold are considered.
    Expands the search if initial neighbors have not commented on the game. Includes the rank of each commenting user based on proximity.

    Args:
    user_id (str): The user ID of the interested user.
    game_title (str): The title of the game for which comments are being retrieved.
    N (int): The number of top comments to return based on relevance.
    word_count_threshold (int): The minimum number of words required for comments to be considered.

    Returns:
    list of tuples: A list containing tuples of (rank, author, comment) if available, or a status message.
    """
    try:
        # Retrieve inner ID of the user
        user_inner_id = trainset.to_inner_uid(user_id)
    except ValueError:
        return f"No data available for the user ID '{user_id}'."

    try:
        # Retrieve inner ID of the game
        game_inner_id = trainset.to_inner_iid(game_title)
    except ValueError:
        return f"No data available for the game '{game_title}'."

    # Initialize variables for searching neighbors and tracking unique comments
    k = 20
    max_neighbors = trainset.n_users  # Maximum possible neighbors
    found_comments = []
    processed_neighbor_ids = set()  # Set to track processed neighbors

    # Retrieve neighbors and expand search until enough comments are found or all users are checked
    while len(found_comments) < N and k <= max_neighbors:
        # Retrieve the k nearest neighbors of the user
        neighbors = algo.get_neighbors(user_inner_id, k=k)
        # Convert inner IDs of the neighbors back to raw IDs and store with ranks
        neighbors_with_ranks = [(rank + 1, trainset.to_raw_uid(inner_id)) for rank, inner_id in enumerate(neighbors) if trainset.to_raw_uid(inner_id) not in processed_neighbor_ids]

        # Filter the dataset to find the neighbors who have rated the specified game
        for rank, neighbor_id in neighbors_with_ranks:
            if neighbor_id not in processed_neighbor_ids:
                processed_neighbor_ids.add(neighbor_id)
                neighbor_comments = data_cleaned[(data_cleaned['author'] == neighbor_id) & (data_cleaned['title'] == game_title)]
                # Apply enhanced word count filter
                valid_comments = filter_comments_by_word_count(neighbor_comments['comment'].tolist(), word_count_threshold)
                for comment in valid_comments:
                    found_comments.append((rank, neighbor_id, comment))

        # Increase the number of neighbors for the next iteration if necessary
        k += 20

    # If comments are found, sort them by length and return the top N
    if found_comments:
        found_comments.sort(key=lambda x: len(x[2]), reverse=True)  # Sort by comment length
        return found_comments[:N]
    else:
        return f"No comments found for the game '{game_title}' from nearest neighbors that meet the word threshold."

# Charger le modèle de langue française
nlp = spacy.load('fr_core_news_sm')

# Fonction adaptée pour traiter chaque commentaire individuellement
def filtrer_commentaire(commentaire):
    doc = nlp(commentaire)
    pos_exclues = ['DET', 'CONJ', 'PRON', 'ADP', 'CCONJ', 'PUNCT']  # Ajout de 'PUNCT' à la liste des exclusions
    #mots_exclus = ['jeu', 'jeux'] and token.lemma_.lower() not in mots_exclus
    # Filtrer les tokens qui ne sont pas dans pos_exclues et dont le lemme n'est pas dans mots_exclus
    mots_filtres = [token.text for token in doc if token.pos_ not in pos_exclues ]
    return ' '.join(mots_filtres)

def tokeniser(commentaire):
    doc = nlp(commentaire)
    tokens = [token.text for token in doc]
    return tokens


def get_top_comments_filtres(user, title, N, word_count_threshold):
    comments = get_top_comments(user, title, N, word_count_threshold)
    new_comments = []
    
    for c in comments:
        c_new=list(c)
        c_new[2] = filtrer_commentaire(c_new[2])
        c_new[2] = tokeniser(c_new[2])
        new_comments.append(c_new)
    
    return new_comments