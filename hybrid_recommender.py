import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from recommender import ContentBasedRecommender

class HybridRecommender(ContentBasedRecommender):
    def __init__(self, movies_df, ratings_df, n_factors=20):
        super().__init__(movies_df, ratings_df)
        
        # Initialize parameters for matrix factorization (SVD)
        self.n_factors = n_factors
        self.svd = None
        self.user_movie_matrix = None
        self.svd_matrix = None
        self._create_collaborative_model()

    def _create_collaborative_model(self):
        # Create a user-item rating matrix for collaborative filtering
        user_movie_matrix = self.ratings_df.pivot_table(index='userId', columns='movieId', values='rating')
        self.user_movie_matrix = user_movie_matrix.fillna(0)
        
        # Perform Singular Value Decomposition (SVD)
        self.svd = TruncatedSVD(n_components=self.n_factors)
        self.svd_matrix = self.svd.fit_transform(self.user_movie_matrix)
        
        # Calculate the cosine similarity matrix for collaborative filtering
        self.collab_similarity_matrix = cosine_similarity(self.svd_matrix)

    def recommend_movies(self, movie_title, top_n=10, model_type='both'):
        # Get content-based recommendations
        content_based_recs = super().recommend_movies(movie_title, top_n=top_n, model_type='both')
        
        # Get collaborative filtering recommendations (user-item matrix similarity)
        movie_id = self.movies_df[self.movies_df['title'] == movie_title]['movieId'].values[0]
        movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
        
        # Get collaborative movie recommendations based on user-item similarity
        movie_similarity_scores = self.collab_similarity_matrix[movie_idx]
        collab_sim_scores = list(enumerate(movie_similarity_scores))
        collab_sim_scores = sorted(collab_sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        collab_recs = [self.movies_df.iloc[i[0]]['title'] for i in collab_sim_scores]
        
        # Combine the content-based and collaborative recommendations
        hybrid_recs = list(set(content_based_recs) | set(collab_recs))
        
        return hybrid_recs[:top_n]

    def recommend_for_user(self, user_id, top_n=10):
        # Get collaborative filtering recommendations for the user
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        user_recommendations = set()
        
        for movie_id in user_ratings['movieId']:
            movie_title = self.movies_df[self.movies_df['movieId'] == movie_id]['title'].values[0]
            similar_movies = self.recommend_movies(movie_title, top_n=top_n, model_type='both')
            user_recommendations.update(similar_movies)
        
        return list(user_recommendations)

# Example of using the Hybrid Recommender
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Initialize the Hybrid Recommender system
hybrid_recommender = HybridRecommender(movies_df, ratings_df)

# Get hybrid recommendations for a sample movie
recommended_movies = hybrid_recommender.recommend_movies('Movie 1', top_n=5)
print("Hybrid Recommended Movies:", recommended_movies)
