class RecommendationSystem:
    def __init__(self, data_path):
        import pandas as pd
        import spacy
        from surprise import Dataset, Reader, KNNWithZScore
        
        # Load language model for French
        self.nlp = spacy.load('fr_core_news_sm')
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        self.algo = None
        self.trainset = None
        self.prepare_data()

    def prepare_data(self):
        """
        Prepares data for recommendation by cleaning and setting up Surprise library structures.
        """
        import pandas as pd
        from surprise import Dataset, Reader
        reader = Reader(rating_scale=(1, 10))
        data_surprise = Dataset.load_from_df(self.data[['author', 'title', 'note']], reader)
        self.trainset = data_surprise.build_full_trainset()

    def train_algorithm(self):
        """
        Trains the KNN with Z-Score algorithm for user-based collaborative filtering.
        """
        from surprise import KNNWithZScore
        sim_options = {'name': 'cosine', 'user_based': True}
        self.algo = KNNWithZScore(sim_options=sim_options, k=20, min_k=1)
        self.algo.fit(self.trainset)

    def filter_comments_by_word_count(self, comments, min_word_count):
        """
        Filters comments by word count.
        """
        return [comment for comment in comments if len(comment.split()) >= min_word_count]


    def get_top_comments(self, user_id, game_title, N, word_count_threshold):
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
            user_inner_id = self.trainset.to_inner_uid(user_id)
        except ValueError:
            return f"No data available for the user ID '{user_id}'."

        try:
            # Retrieve inner ID of the game
            game_inner_id = self.trainset.to_inner_iid(game_title)
        except ValueError:
            return f"No data available for the game '{game_title}'."

        # Initialize variables for searching neighbors and tracking unique comments
        k = 20
        max_neighbors = self.trainset.n_users  # Maximum possible neighbors
        found_comments = []
        processed_neighbor_ids = set()  # Set to track processed neighbors

        # Retrieve neighbors and expand search until enough comments are found or all users are checked
        while len(found_comments) < N and k <= max_neighbors:
            # Retrieve the k nearest neighbors of the user
            neighbors = self.algo.get_neighbors(user_inner_id, k=k)
            # Convert inner IDs of the neighbors back to raw IDs and store with ranks
            neighbors_with_ranks = [(rank + 1, self.trainset.to_raw_uid(inner_id)) for rank, inner_id in enumerate(neighbors) if self.trainset.to_raw_uid(inner_id) not in processed_neighbor_ids]

            # Filter the dataset to find the neighbors who have rated the specified game
            for rank, neighbor_id in neighbors_with_ranks:
                if neighbor_id not in processed_neighbor_ids:
                    processed_neighbor_ids.add(neighbor_id)
                    neighbor_comments = self.data[(self.data['author'] == neighbor_id) & (self.data['title'] == game_title)]
                    # Apply enhanced word count filter
                    valid_comments = self.filter_comments_by_word_count(neighbor_comments['comment'].tolist(), word_count_threshold)
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
        
    def filtrer_commentaire(self, commentaire):
        """
        Filters tokens from a comment based on specific POS tags.
        """
        doc = self.nlp(commentaire)
        pos_exclues = ['DET', 'CONJ', 'PRON', 'ADP', 'CCONJ', 'PUNCT']
        mots_filtres = [token.text for token in doc if token.pos_ not in pos_exclues]
        return ' '.join(mots_filtres)

    def tokeniser(self, commentaire):
        """
        Tokenizes a comment into a list of words.
        """
        doc = self.nlp(commentaire)
        return [token.text for token in doc]

    def get_top_comments_filtres(self, user, title, N, word_count_threshold):
        """
        Retrieves and processes top comments using NLP techniques.
        """
        comments = self.get_top_comments(user, title, N, word_count_threshold)
        new_comments = []
        
        for c in comments:
            c_new = list(c)
            c_new[2] = self.filtrer_commentaire(c_new[2])
            c_new[2] = self.tokeniser(c_new[2])
            new_comments.append(c_new)
        
        return new_comments

    def explication_gpt(self, key, user, title, N, word_count_threshold):
        """
        Generates an explanation for recommendations using GPT from OpenAI.
        """
        import openai
        client = openai.OpenAI(api_key=key)
        
        comments = self.get_top_comments_filtres(user, title, N, word_count_threshold)
        prompt = "Je veux recommander un jeu a un joueur, apres avoir fait tourné un algorithme, voici ce que ses k plus proches voisins ont commenté dessus (les commentaires ont été filtré par des techniques de NLP et chaque commentaire est de la forme (rang dans la liste des k plus proches voisins, auteur, commentaire)): "+ str(comments) +" Je veux que tu me génère une explication de la recommandation grace a ces commentaires."
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-3.5-turbo"
        )
        
        return chat_completion.choices[0].message.content


# recommender = RecommendationSystem("BDD/avis_sans_outliers.csv")
# recommender.train_algorithm()
# explanation = recommender.explication_gpt("api_key", "user123", "Some Game", 5, 4)
