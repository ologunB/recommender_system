from data import generate_movies_data, generate_ratings_data
from recommender import ContentBasedRecommender

# Generate synthetic data
movies_df = generate_movies_data(num_movies=100)
ratings_df = generate_ratings_data(movies_df, num_users=50)

# Initialize the recommender system
recommender = ContentBasedRecommender(movies_df, ratings_df)

# Recommend movies for a sample user
user_id = 1
recommended_movies = recommender.recommend_for_user(user_id, top_n=5)

print(f"Recommended movies for User {user_id}:")
for movie in recommended_movies:
    print(movie)
