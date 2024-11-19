import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, movies_df, ratings_df):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.tfidf_matrix_genres = None
        self.tfidf_matrix_descriptions = None
        self.cosine_sim_genres = None
        self.cosine_sim_descriptions = None
        self._create_model()

    def _create_model(self):
        # Genre-based TF-IDF
        tfidf_genres = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix_genres = tfidf_genres.fit_transform(self.movies_df['genres'])

        # Description-based TF-IDF
        tfidf_desc = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix_descriptions = tfidf_desc.fit_transform(self.movies_df['description'])

        # Calculate cosine similarity for both genres and descriptions
        self.cosine_sim_genres = cosine_similarity(self.tfidf_matrix_genres, self.tfidf_matrix_genres)
        self.cosine_sim_descriptions = cosine_similarity(self.tfidf_matrix_descriptions, self.tfidf_matrix_descriptions)

    def recommend_movies(self, movie_title, top_n=10, model_type='both'):
        if movie_title not in self.movies_df['title'].values:
            raise ValueError(f"Movie '{movie_title}' not found in the dataset.")
        
        # Get the index of the movie that matches the title
        idx = self.movies_df.index[self.movies_df['title'] == movie_title].tolist()[0]
        
        # Get similarity scores based on the selected model (genres, descriptions, or both)
        if model_type == 'genres':
            sim_scores = list(enumerate(self.cosine_sim_genres[idx]))
        elif model_type == 'descriptions':
            sim_scores = list(enumerate(self.cosine_sim_descriptions[idx]))
        else:  # Combine both genre and description similarities
            sim_scores = [(i, (self.cosine_sim_genres[idx][i] + self.cosine_sim_descriptions[idx][i]) / 2) for i in range(len(self.movies_df))]

        # Sort the movies based on the similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the top n most similar movies
        sim_scores = sim_scores[1:top_n + 1]
        movie_indices = [i[0] for i in sim_scores]
        
        # Return movie titles of the most similar movies
        return self.movies_df['title'].iloc[movie_indices].tolist()

    def recommend_for_user(self, user_id, top_n=10):
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        recommendations = set()
        
        for movie_id in user_ratings['movieId']:
            movie_title = self.movies_df[self.movies_df['movieId'] == movie_id]['title'].values[0]
            similar_movies = self.recommend_movies(movie_title, top_n)
            recommendations.update(similar_movies)
        
        return list(recommendations)

# Example of creating and using the recommender system
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
recommender = ContentBasedRecommender(movies_df, ratings_df)

# Recommend movies based on genre and description
recommended_movies = recommender.recommend_movies('Movie 1', top_n=5, model_type='both')
print("Recommended movies:", recommended_movies)


