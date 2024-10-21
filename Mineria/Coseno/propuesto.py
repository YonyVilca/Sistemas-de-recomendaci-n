import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import time
import psutil
import os

# Inicia el monitoreo de rendimiento
process = psutil.Process(os.getpid())
start_time = time.time()

# Paso 1: Función para cargar el dataset de calificaciones desde un archivo .csv
def load_movielens(file_path):
    """
    Cargar el dataset MovieLens desde un archivo .csv con la estructura: userId,movieId,rating,timestamp.
    
    Parámetros:
    - file_path: Ruta del archivo de calificaciones .csv
    
    Retorna:
    - DataFrame de calificaciones
    """
    # Leer archivo .csv
    ratings = pd.read_csv(file_path)
    
    # Retornar solo las columnas relevantes
    return ratings[['userId', 'movieId', 'rating']]

# Paso 2: Función para cargar los nombres de las películas desde un archivo .csv
def load_movie_titles(movie_file_path):
    """
    Cargar los nombres de las películas desde un archivo .csv con la estructura: movieId,title,genres.
    
    Parámetros:
    - movie_file_path: Ruta del archivo de películas
    
    Retorna:
    - DataFrame con movieId y title
    """
    # Leer archivo .csv
    movies = pd.read_csv(movie_file_path)
    
    # Retornar solo movieId y title
    return movies[['movieId', 'title']]

# Paso 3: Cargar los datasets de calificaciones y películas
file_path_ratings = '/workspaces/Paralela/Mineria/Coseno/ratings.csv'  # Ruta al archivo .csv de calificaciones
file_path_movies = '/workspaces/Paralela/Mineria/Coseno/movies.csv'   # Ruta al archivo .csv de películas

# Cargar las calificaciones
ratings = load_movielens(file_path_ratings)

# Cargar los nombres de las películas
movies = load_movie_titles(file_path_movies)

# Paso 4: Configurar el lector de Surprise
reader = Reader(rating_scale=(1, 5))

# Cargar los datos en formato que Surprise pueda manejar
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Paso 5: Dividir los datos en conjunto de entrenamiento y prueba
trainset, testset = train_test_split(data, test_size=0.25)

# Paso 6: Inicializar el modelo SVD
model = SVD()

# Paso 7: Entrenar el modelo con el conjunto de entrenamiento
model.fit(trainset)

# Paso 8: Hacer predicciones sobre el conjunto de prueba
predictions = model.test(testset)

# Paso 9: Evaluar el rendimiento con RMSE (Root Mean Squared Error)
print("Evaluación del modelo SVD:")
accuracy.rmse(predictions)

# Paso 10: Predecir la calificación de un usuario para un ítem específico
user_id = 6  # ID del usuario de ejemplo
movie_id = 108932  # ID de la película de ejemplo

# Usar el modelo para predecir
prediction = model.predict(user_id, movie_id)

# Vincular el movieId con el nombre de la película
movie_name = movies[movies['movieId'] == movie_id]['title'].values[0]

# Mostrar el resultado con el nombre de la película
print(f"Predicción para el usuario {user_id} y la película '{movie_name}': {prediction.est:.4f}")

# Fin del monitoreo de rendimiento
end_time = time.time()
execution_time = end_time - start_time
memory_use = process.memory_info().rss / (1024 ** 2)  # Memoria en MB
cpu_percent = process.cpu_percent()

print(f"\nTiempo de ejecución total: {execution_time:.2f} segundos")
print(f"Uso de memoria: {memory_use:.2f} MB")
print(f"Uso de CPU: {cpu_percent}%")

# Medir el tamaño de los archivos de datos
ratings_file_size = os.path.getsize(file_path_ratings) / (1024 ** 2)  # Tamaño en MB
movies_file_size = os.path.getsize(file_path_movies) / (1024 ** 2)  # Tamaño en MB

print(f"Tamaño del archivo de ratings: {ratings_file_size:.2f} MB")
print(f"Tamaño del archivo de películas: {movies_file_size:.2f} MB")
