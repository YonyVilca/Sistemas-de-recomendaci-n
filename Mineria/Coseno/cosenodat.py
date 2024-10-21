import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time
import psutil
import os

# Inicia el monitoreo de rendimiento
process = psutil.Process(os.getpid())
start_time = time.time()

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

    def normalize_ratings(self, ratings_matrix):
        """ Normalize user ratings to the range [-1, 1]. """
        return 2 * (ratings_matrix - self.min_rating) / (self.max_rating - self.min_rating) - 1

    def denormalize_rating(self, normalized_rating):
        """ Denormalize a normalized rating back to its original scale. """
        return 0.5 * ((normalized_rating + 1) * (self.max_rating - self.min_rating)) + self.min_rating

    def adjusted_cosine_similarity(self, ratings_matrix):
        """ Calculate adjusted cosine similarity between items. """
        normalized_ratings = self.normalize_ratings(ratings_matrix)
        user_mean = normalized_ratings.mean(axis=1)
        adjusted_ratings = normalized_ratings.sub(user_mean, axis=0)
        
        # Convert to sparse matrix
        sparse_ratings = csr_matrix(adjusted_ratings.fillna(0))
        
        # Calculate cosine similarity using sparse matrix
        similarity_matrix = cosine_similarity(sparse_ratings.T, dense_output=False)
        
        return pd.DataFrame(similarity_matrix.toarray(), index=ratings_matrix.columns, columns=ratings_matrix.columns)
    
    def predict(self, user, item, similarity_matrix):
        """ Predict the rating of a user for an unrated item. """
        rated_items = self.data.loc[user].dropna().index
        similarities = similarity_matrix.loc[item, rated_items]
        normalized_ratings = self.normalize_ratings(self.data.loc[user, rated_items])
        
        numerator = np.sum(similarities * normalized_ratings)
        denominator = np.sum(np.abs(similarities))
        
        normalized_prediction = numerator / denominator if denominator != 0 else 0
        return self.denormalize_rating(normalized_prediction)

# Cargar el archivo ratings.dat (sin cabecera y delimitado por "::")
ratings = pd.read_csv('/workspaces/Paralela/Mineria/Coseno/ratings10m.dat', sep='::', header=None, engine='python', names=['userId', 'movieId', 'rating', 'timestamp'], encoding='latin1')

# Cargar el archivo movies.dat (sin cabecera y delimitado por "::")
movies = pd.read_csv('/workspaces/Paralela/Mineria/Coseno/movies10m.dat', sep='::', header=None, engine='python', names=['movieId', 'title', 'genres'], encoding='latin1')

# Reemplazar calificaciones 0 con NaN para indicar ítems no calificados
ratings['rating'] = ratings['rating'].replace(0, np.nan)

# Pivotar el DataFrame de ratings para crear una matriz usuario-ítem
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Crear el sistema de recomendación
recommender_system = Recommender(user_item_matrix, metric='adjusted_cosine')

# Calcular la matriz de similitud entre ítems
similarity_matrix = recommender_system.adjusted_cosine_similarity(user_item_matrix)

# Predecir la calificación para un usuario y película específicos
user_id = 1  # ID de usuario de ejemplo
movie_id = 15   # ID de película de ejemplo
prediction = recommender_system.predict(user_id, movie_id, similarity_matrix)

# Obtener el nombre de la película
movie_name = movies.loc[movies['movieId'] == movie_id, 'title'].values[0]

# Imprimir la predicción con más decimales y el nombre de la película
print(f"Predicción para el usuario {user_id} y la película '{movie_name}': {prediction:.6f}")

# Fin del monitoreo de rendimiento
end_time = time.time()
execution_time = end_time - start_time
memory_use = process.memory_info().rss / (1024 ** 2)  # Memoria en MB
cpu_percent = process.cpu_percent()

print(f"\nTiempo de ejecución total: {execution_time:.2f} segundos")
print(f"Uso de memoria: {memory_use:.2f} MB")
print(f"Uso de CPU: {cpu_percent}%")

# Medir el tamaño de los archivos de datos
ratings_file_size = os.path.getsize('/workspaces/Paralela/Mineria/Coseno/ratings1m.dat') / (1024 ** 2)  # Tamaño en MB
movies_file_size = os.path.getsize('/workspaces/Paralela/Mineria/Coseno/movies1m.dat') / (1024 ** 2)  # Tamaño en MB

print(f"Tamaño del archivo de ratings: {ratings_file_size:.2f} MB")
print(f"Tamaño del archivo de películas: {movies_file_size:.2f} MB")

# Medir el tamaño de las matrices en memoria
user_item_matrix_size = user_item_matrix.memory_usage(deep=True).sum() / (1024 ** 2)  # Tamaño en MB
similarity_matrix_size = similarity_matrix.memory_usage(deep=True).sum() / (1024 ** 2)  # Tamaño en MB

print(f"Tamaño de la matriz usuario-ítem: {user_item_matrix_size:.2f} MB")
print(f"Tamaño de la matriz de similitud: {similarity_matrix_size:.2f} MB")
