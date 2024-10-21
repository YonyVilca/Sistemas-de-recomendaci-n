import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

class Recommender:
    def __init__(self, data, metric='adjusted_cosine', max_rating=5, min_rating=1):
        """ Initialize recommender
        metric is which distance formula to use"""
        self.metric = metric
        self.max_rating = max_rating
        self.min_rating = min_rating
        
        # Convert data to DataFrame if it's a dictionary
        if isinstance(data, dict):
            self.data = pd.DataFrame(data).T
        else:
            self.data = data

        if self.metric == 'adjusted_cosine':
            self.fn = self.adjusted_cosine_similarity

        # Create user-item matrix
        self.user_item_matrix = self.data.pivot(index='userId', columns='movieId', values='rating').fillna(0)

        # Convert user-item matrix to sparse matrix
        self.user_item_sparse_matrix = csr_matrix(self.user_item_matrix.values)

        # Calculate similarity matrix
        self.similarity_matrix = self.fn(self.user_item_matrix)

        # Perform SVD
        self.U, self.sigma, self.Vt = svds(self.user_item_sparse_matrix, k=50)
        self.sigma = np.diag(self.sigma)
        self.svd_predictions = np.dot(np.dot(self.U, self.sigma), self.Vt)
        self.svd_predictions_df = pd.DataFrame(self.svd_predictions, columns=self.user_item_matrix.columns)

        # Train linear regression model
        self.model = LinearRegression()
        self.train_model()

    def adjusted_cosine_similarity(self, matrix):
        # Mean center the ratings
        mean_user_rating = matrix.mean(axis=1)
        ratings_diff = (matrix.T - mean_user_rating).T

        # Compute the cosine similarity
        similarity = ratings_diff.dot(ratings_diff.T) / np.sqrt(np.outer(np.square(ratings_diff).sum(axis=1), np.square(ratings_diff).sum(axis=1)))
        return similarity

    def normalize_ratings(self, ratings_matrix):
        """ Normalize user ratings to the range [-1, 1]. """
        return 2 * (ratings_matrix - self.min_rating) / (self.max_rating - self.min_rating) - 1

    def denormalize_rating(self, normalized_rating):
        """ Denormalize ratings from the range [-1, 1] to the original range. """
        return normalized_rating * (self.max_rating - self.min_rating) / 2 + (self.max_rating + self.min_rating) / 2

    def train_model(self):
        """ Train a linear regression model on the user-item matrix. """
        ratings = self.data.dropna()
        X_train = ratings[['userId', 'movieId']]
        y_train = ratings['rating']
        self.model.fit(X_train, y_train)

    def predict(self, user_id, item_id):
        """ Predict rating for a specific user and item using hybrid approach. """
        # Slope One prediction (placeholder, implement actual Slope One logic if needed)
        slope_one_pred = self.svd_predictions_df.loc[user_id, item_id]

        # Adjusted Cosine Similarity prediction
        cosine_pred = self.similarity_matrix.loc[user_id, item_id]

        # SVD prediction
        svd_pred = self.svd_predictions_df.loc[user_id, item_id]

        # Linear regression model prediction
        model_pred = self.model.predict([[user_id, item_id]])[0]

        # Combine predictions
        final_prediction = (slope_one_pred + cosine_pred + svd_pred + model_pred) / 4
        return self.denormalize_rating(final_prediction)

# Load the ratings dataset
ratings = pd.read_csv('/workspaces/Paralela/Mineria/Coseno/ratings.csv')

# Create recommender system
recommender_system = Recommender(ratings, metric='adjusted_cosine')

# Predict rating for a specific user and item
user_id = 606  # Example user ID
movie_id = 10  # Example movie ID
prediction = recommender_system.predict(user_id, movie_id)

# Print the denormalized prediction
print(f"Predicción para el usuario {user_id} y la película {movie_id}: {prediction:.6f}")