import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("spotify_tracks.csv")
df.dropna(subset=['track_name', 'artist', 'album_name'], inplace=True)
def create_feature_soup(row):
    return f"{row['track_name']} {row['artist']} {row['album_name']} {row['release_date']}"
df['soup'] = df.apply(create_feature_soup, axis=1)

# Vectorize the text data and compute cosine similarity
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['soup'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_songs(track_index, num_recommendations=5):
    sim_scores = list(enumerate(cosine_sim[track_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    recommended_indices = [i[0] for i in sim_scores]
    return df.iloc[recommended_indices][['track_name', 'artist', 'album_name', 'release_date']]

st.title("Spotify Music Recommendation System")

# Create a dropdown for track selection
track_list = df['track_name'].tolist()
selected_track = st.selectbox("Select a track:", track_list)
selected_index = df[df['track_name'] == selected_track].index[0]

if st.button("Get Recommendations"):
    recommendations = recommend_songs(selected_index, num_recommendations=5)
    st.write("Recommended Tracks:")
    st.dataframe(recommendations)
