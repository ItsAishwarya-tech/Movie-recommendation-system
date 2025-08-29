import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("genre_movies.csv")
# Drop duplicate rows (same Movie Name & Summary)
df = df.drop_duplicates(subset=["Movie Name", "Summary"], keep="first").reset_index(drop=True)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["Summary"].fillna(""))

#Function to recommend movies
def recommend_movies(user_input, top_n=5):
    # Transform user input into vector
    user_vec = vectorizer.transform([user_input])
    
    # Compute cosine similarity with dataset
    similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()
    
    # Get top N similar movies
    top_indices = similarity.argsort()[-top_n:][::-1]
    
    results = df.iloc[top_indices][["Movie Name", "Summary"]]
    results["Score"] = similarity[top_indices]
    return results

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Enter a short movie storyline and get top recommended movies based on similarity!")

# User input
user_input = st.text_area("Enter movie storyline here:")

if st.button("Get Recommendations"):
    if user_input.strip() == "":
        st.warning("Please enter a storyline to get recommendations.")
    else:
        recommendations = recommend_movies(user_input)
        st.subheader("Top Recommendations:")
        for i, row in recommendations.iterrows():
            st.markdown(f"**{row['Movie Name']}**  (Score: {row['Score']:.3f})")
            st.write(row["Summary"])
            st.write("---")




