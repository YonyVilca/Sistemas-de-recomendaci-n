import pandas as pd
import numpy as np
from numba import jit
import time
import psutil
import os

# Inicia el monitoreo de rendimiento
process = psutil.Process(os.getpid())
start_time = time.time()

# Ruta al archivo .dat de MovieLens Small
file_path = '/workspaces/Paralela/Mineria/Coseno/ratings10m.dat'
movies_file_path = '/workspaces/Paralela/Mineria/Coseno/movies10m.dat'

# Cargar el archivo de ratings .dat (sin cabecera y delimitado por "::")
ratings_df = pd.read_csv(file_path, sep='::', header=None, engine='python', names=['user', 'item', 'rating', 'timestamp'], encoding='latin1')

# Cargar el archivo de nombres de películas .dat (sin cabecera y delimitado por "::")
movies_df = pd.read_csv(movies_file_path, sep='::', header=None, engine='python', names=['movieId', 'title', 'genres'], encoding='latin1')

# Crear un diccionario para mapear IDs de películas a nombres
movie_names = {row['movieId']: row['title'] for _, row in movies_df.iterrows()}

# Pivotar el DataFrame para crear una matriz de calificaciones de usuarios por ítems
ratings_df.columns = ['user', 'item', 'rating', 'timestamp']
ratings_matrix = ratings_df.pivot(index='user', columns='item', values='rating')

# Convertir la matriz de ratings a un array de numpy para facilitar el cálculo con numba
ratings_matrix_np = ratings_matrix.to_numpy()

# Convertir índices de usuarios y columnas de películas a listas para acceder con numpy
users = ratings_matrix.index.to_numpy()
items = ratings_matrix.columns.to_numpy()

# Diccionarios para mapear entre índices de usuarios y columnas
user_map = {user: idx for idx, user in enumerate(users)}
item_map = {item: idx for idx, item in enumerate(items)}

# Función para calcular las desviaciones
@jit(nopython=True)
def compute_deviations_np(ratings):
    num_items = ratings.shape[1]
    deviations = np.zeros((num_items, num_items), dtype=np.float64)
    frequencies = np.zeros((num_items, num_items), dtype=np.int32)

    for user_ratings in ratings:
        for item1 in range(num_items):
            if not np.isnan(user_ratings[item1]):
                for item2 in range(num_items):
                    if item1 != item2 and not np.isnan(user_ratings[item2]):
                        deviations[item1, item2] += user_ratings[item1] - user_ratings[item2]
                        frequencies[item1, item2] += 1

    for item1 in range(num_items):
        for item2 in range(num_items):
            if frequencies[item1, item2] > 0:
                deviations[item1, item2] /= frequencies[item1, item2]

    return deviations, frequencies

# Función para predecir la calificación utilizando Slope One
@jit(nopython=True)
def slope_one_prediction_np(user_idx, item_idx, ratings, deviations, frequencies):
    numerator = 0.0
    denominator = 0

    for other_item_idx in range(ratings.shape[1]):
        if other_item_idx != item_idx and not np.isnan(ratings[user_idx, other_item_idx]):
            if frequencies[item_idx, other_item_idx] > 0:
                weight = frequencies[item_idx, other_item_idx]
                numerator += (deviations[item_idx, other_item_idx] + ratings[user_idx, other_item_idx]) * weight
                denominator += weight

    if denominator != 0:
        return numerator / denominator
    else:
        return np.nan

# Calcular desviaciones y frecuencias con Numba y Numpy
deviations_np, frequencies_np = compute_deviations_np(ratings_matrix_np)

# Elegir un usuario y una película para realizar la predicción
user_id = 1
item_id = 15

# Obtener índices para numpy basados en user_id y item_id
user_idx = user_map[user_id]
item_idx = item_map[item_id]

# Realizar la predicción para el usuario y película específicos
predicted_rating_np = slope_one_prediction_np(user_idx, item_idx, ratings_matrix_np, deviations_np, frequencies_np)

# Mostrar la calificación predicha con el nombre de la película
movie_name = movie_names.get(item_id, "Desconocido")
print(f"\nCalificación predicha del usuario {user_id} en la película '{movie_name}': {predicted_rating_np:.3f}")

# Fin del monitoreo de rendimiento
end_time = time.time()
execution_time = end_time - start_time
memory_use = process.memory_info().rss / (1024 ** 2)  # Memoria en MB
cpu_percent = process.cpu_percent()

print(f"\nTiempo de ejecución total: {execution_time:.2f} segundos")
print(f"Uso de memoria: {memory_use:.2f} MB")
print(f"Uso de CPU: {cpu_percent}%")

# Medir el tamaño de los archivos de datos
ratings_file_size = os.path.getsize(file_path) / (1024 ** 2)  # Tamaño en MB
movies_file_size = os.path.getsize(movies_file_path) / (1024 ** 2)  # Tamaño en MB

print(f"Tamaño del archivo de ratings: {ratings_file_size:.2f} MB")
print(f"Tamaño del archivo de películas: {movies_file_size:.2f} MB")

# Medir el tamaño de las matrices en memoria
ratings_matrix_size = ratings_matrix_np.nbytes / (1024 ** 2)  # Tamaño en MB
deviations_size = deviations_np.nbytes / (1024 ** 2)  # Tamaño en MB
frequencies_size = frequencies_np.nbytes / (1024 ** 2)  # Tamaño en MB

print(f"Tamaño de la matriz de calificaciones: {ratings_matrix_size:.2f} MB")
print(f"Tamaño de la matriz de desviaciones: {deviations_size:.2f} MB")
print(f"Tamaño de la matriz de frecuencias: {frequencies_size:.2f} MB")
