import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt # type: ignore

def generate_movies_data(num_movies=100):
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Documentary', 'Animation']
    movie_ids = range(1, num_movies + 1)
    movie_titles = [f'Movie {i}' for i in movie_ids]
    movie_genres = [', '.join(random.sample(genres, random.randint(1, 3))) for _ in range(num_movies)]
    movie_descriptions = [f"Description of Movie {i}" for i in movie_ids]  # Added descriptions
    
    movies_df = pd.DataFrame({
        'movieId': movie_ids,
        'title': movie_titles,
        'genres': movie_genres,
        'description': movie_descriptions
    })
    
    return movies_df

def generate_ratings_data(movies_df, num_users=50):
    user_ids = range(1, num_users + 1)
    ratings_data = []
    
    for user_id in user_ids:
        movie_ids = random.sample(movies_df['movieId'].tolist(), random.randint(5, 20))
        for movie_id in movie_ids:
            rating = random.randint(1, 5)
            ratings_data.append([user_id, movie_id, rating])
    
    ratings_df = pd.DataFrame(ratings_data, columns=['userId', 'movieId', 'rating'])
    return ratings_df

def data_exploration(movies_df, ratings_df):
    # Visualize the rating distribution
    plt.figure(figsize=(10, 6))
    ratings_df['rating'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.show()

    # Visualize the genre distribution
    genre_counts = movies_df['genres'].str.split(',').explode().value_counts()
    genre_counts.plot(kind='bar', color='orange', figsize=(10, 6))
    plt.title('Distribution of Movie Genres')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.show()

# Generate synthetic data and explore
movies_df = generate_movies_data(num_movies=100)
ratings_df = generate_ratings_data(movies_df, num_users=50)

data_exploration(movies_df, ratings_df)

# Save the data
movies_df.to_csv('movies.csv', index=False)
ratings_df.to_csv('ratings.csv', index=False)
