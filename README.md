
# Movie Recommendation System

This repository contains a movie recommendation system built using both **content-based** and **collaborative filtering** techniques. It allows for recommending movies based on their genres, descriptions, and user preferences. The project also includes a **hybrid recommender** that combines content-based and collaborative filtering methods.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Data Generation](#data-generation)
- [Recommendation Algorithms](#recommendation-algorithms)
  - [Content-Based Recommender](#content-based-recommender)
  - [Hybrid Recommender](#hybrid-recommender)
- [Usage](#usage)
- [Example Usage](#example-usage)
- [License](#license)

---

## Overview

This project implements a movie recommendation system using two main approaches:

1. **Content-Based Filtering**: Recommends movies based on the similarities between the content (genres and descriptions) of movies that a user has liked.
2. **Collaborative Filtering**: Recommends movies based on similarities between users and their past movie ratings.
3. **Hybrid Recommender**: Combines both content-based and collaborative filtering to provide recommendations, taking advantage of both techniques.

The system generates synthetic movie data with various genres and descriptions and rating data for users. The recommender uses **TF-IDF** for textual data (genres and descriptions) and **Singular Value Decomposition (SVD)** for collaborative filtering.

---

## Project Structure

The project contains the following Python scripts:

- `data.py`: Contains functions for generating synthetic movie data and ratings data.
- `recommender.py`: Implements the content-based movie recommender.
- `hybrid_recommender.py`: Implements the hybrid recommender that combines content-based and collaborative filtering.
- `main.py`: Initializes the system, generates data, and demonstrates how to get recommendations.
- `requirements.txt`: Lists the required Python packages for the project.

---

## Requirements

To run the project, you need Python 3.x and the following dependencies:

- pandas
- numpy
- matplotlib
- scikit-learn

You can install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

---

## Data Generation

The `data.py` script contains two main functions for generating synthetic movie and ratings data:

### `generate_movies_data(num_movies=100)`

Generates a DataFrame containing `num_movies` movies. Each movie has:

- A unique `movieId`.
- A random title in the form "Movie X".
- A random set of genres (1-3 genres per movie).
- A description of the movie.

### `generate_ratings_data(movies_df, num_users=50)`

Generates a DataFrame containing movie ratings from `num_users` users. Each user has rated between 5 and 20 random movies with a rating between 1 and 5.

---

## Recommendation Algorithms

### Content-Based Recommender

The `ContentBasedRecommender` class in `recommender.py` uses **TF-IDF** to represent movie genres and descriptions as vectors and calculates the **cosine similarity** between them. Based on the selected model (genres, descriptions, or both), it recommends similar movies.

#### Methods:

- **`__init__(self, movies_df, ratings_df)`**: Initializes the recommender with movie and ratings data.
- **`_create_model(self)`**: Creates the content-based model using TF-IDF and calculates the cosine similarity.
- **`recommend_movies(self, movie_title, top_n=10, model_type='both')`**: Recommends `top_n` similar movies based on the specified model type (`'genres'`, `'descriptions'`, or `'both'`).
- **`recommend_for_user(self, user_id, top_n=10)`**: Recommends movies for a user based on their movie ratings.

### Hybrid Recommender

The `HybridRecommender` class in `hybrid_recommender.py` extends the `ContentBasedRecommender` and adds collaborative filtering based on **Singular Value Decomposition (SVD)**. It combines both content-based and collaborative recommendations for better results.

#### Methods:

- **`__init__(self, movies_df, ratings_df, n_factors=20)`**: Initializes the hybrid recommender with movie and ratings data and performs matrix factorization (SVD).
- **`_create_collaborative_model(self)`**: Creates the collaborative filtering model using SVD and calculates the cosine similarity for collaborative filtering.
- **`recommend_movies(self, movie_title, top_n=10, model_type='both')`**: Recommends movies by combining content-based and collaborative filtering.
- **`recommend_for_user(self, user_id, top_n=10)`**: Recommends movies for a user by combining content-based and collaborative recommendations.

---

## Usage

1. **Generate Data**: The data can be generated using the functions in `data.py`. You can modify the number of movies and users as required.
2. **Create Recommender**: You can initialize the content-based recommender or hybrid recommender by loading the generated data.
3. **Get Recommendations**: Once the recommender is created, you can get movie recommendations either by specifying a movie or a user.

---

## Example Usage

1. **Generating Data and Exploring**:
   In `main.py`, the following code demonstrates how to generate movie data, create the recommender system, and get recommendations:

```python
from data import generate_movies_data, generate_ratings_data
from recommender import ContentBasedRecommender

# Generate synthetic data
movies_df = generate_movies_data(num_movies=100)
ratings_df = generate_ratings_data(movies_df, num_users=50)

# Initialize the recommender system
recommender = ContentBasedRecommender(movies_df, ratings_df)

# Recommend movies based on genre and description
recommended_movies = recommender.recommend_movies('Movie 1', top_n=5, model_type='both')
print("Recommended movies:", recommended_movies)

# Recommend movies for a sample user
user_id = 1
recommended_movies_user = recommender.recommend_for_user(user_id, top_n=5)
print(f"Recommended movies for User {user_id}: {recommended_movies_user}")
```

2. **Using the Hybrid Recommender**:
   The following code demonstrates how to use the hybrid recommender for getting movie recommendations:

```python
from hybrid_recommender import HybridRecommender
import pandas as pd

# Load the generated data
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Initialize the Hybrid Recommender system
hybrid_recommender = HybridRecommender(movies_df, ratings_df)

# Get hybrid recommendations for a sample movie
recommended_movies = hybrid_recommender.recommend_movies('Movie 1', top_n=5)
print("Hybrid Recommended Movies:", recommended_movies)

# Get hybrid recommendations for a user
user_id = 1
recommended_movies_user = hybrid_recommender.recommend_for_user(user_id, top_n=5)
print(f"Recommended movies for User {user_id}: {recommended_movies_user}")
```

---

## License

This project is licensed under the MIT License.
