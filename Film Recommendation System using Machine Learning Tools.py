import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
print('Project Topic : Film recommendation system using machine learning tools')
print('AI Internship by Codespectra')
print('By Vipul Pandey')

# Load MovieLens dataset (or your own dataset)
data = Dataset.load_builtin('ml-100k')

# Define the reader for the dataset
reader = Reader(rating_scale=(1, 5))

# Load the dataset into Surprise format
dataset = data.build_full_trainset()

# Split the dataset into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.2)

# Build the user-based collaborative filtering model
sim_options = {
    'name': 'cosine',
    'user_based': True
}
cf_model = KNNBasic(sim_options=sim_options)
cf_model.fit(trainset)

# Make predictions on the test set
cf_predictions = cf_model.test(testset)

# Load movie information (genres) for content-based filtering
movies = pd.read_csv('movies.csv')  # Assuming you have a CSV file with movie information

# Create a TF-IDF matrix for content-based filtering
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'].fillna(''))

# Calculate cosine similarity between movies based on genres
content_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get content-based recommendations for a movie
def get_content_recommendations(movie_id, top_n=10):
    movie_index = movies.index[movies['movieId'] == movie_id].tolist()[0]
    sim_scores = list(enumerate(content_similarities[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    movie_indices = [x[0] for x in sim_scores]
    return movies['movieId'].iloc[movie_indices]

# Function to get hybrid recommendations for a user
def get_hybrid_recommendations(user_id, top_n=10):
    # Get collaborative filtering recommendations
    cf_user_predictions = [cf_model.predict(user_id, movie_id) for movie_id in dataset.all_items()]
    cf_sorted_predictions = sorted(cf_user_predictions, key=lambda x: x.est, reverse=True)

    # Get the top-rated movie from collaborative filtering
    top_cf_movie_id = cf_sorted_predictions[0].iid

    # Get content-based recommendations for the top-rated movie
    content_recommendations = get_content_recommendations(top_cf_movie_id, top_n)

    return content_recommendations

# Example usage
user_id = str(1)
hybrid_recommendations = get_hybrid_recommendations(user_id)
print("Hybrid Recommendations:", hybrid_recommendations.tolist())
