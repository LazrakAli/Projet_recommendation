import pandas as pd

def filter_reviews(dataframe):
    """
    Filters the dataframe to keep only games with at least 10 reviews and authors who posted at least 10 reviews.
    This process is repeated until no more rows are removed.
    
    Args:
    dataframe (pd.DataFrame): The input dataframe containing game reviews.
    
    Returns:
    pd.DataFrame: The filtered dataframe.
    """
    previous_length = None
    current_length = len(dataframe)

    while previous_length != current_length:
        # Update the previous length for comparison in the next iteration
        previous_length = current_length

        # Filter games with at least 10 reviews
        game_counts = dataframe['title'].value_counts()
        games_with_enough_reviews = game_counts[game_counts >= 10].index
        dataframe = dataframe[dataframe['title'].isin(games_with_enough_reviews)]

        # Filter authors with at least 10 reviews
        author_counts = dataframe['author'].value_counts()
        authors_with_enough_reviews = author_counts[author_counts >= 10].index
        dataframe = dataframe[dataframe['author'].isin(authors_with_enough_reviews)]

        # Update the current length after filtering
        current_length = len(dataframe)

    return dataframe
