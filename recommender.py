import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("filtered_books_by_black_authors.csv")

# Load precomputed embeddings
embeddings = np.load("book_embeddings.npy")

# Load the pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def recommend_books(query, top_n=5):
    """
    Given a user query, find the most similar books based on description embeddings.
    """
    # Encode the user query
    query_embedding = model.encode([query])

    # Compute cosine similarity between the query and all book embeddings
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Get the indices of the top N most similar books
    top_indices = similarities.argsort()[-top_n:][::-1]

    # Display recommended books
    recommendations = []
    for i, index in enumerate(top_indices):
        book_info = {
            "Rank": i + 1,
            "Title": df.iloc[index]['Title'],
            "Author": df.iloc[index]['Authors'],
            "Description": df.iloc[index]['Description'][:300]  # Show first 300 characters
        }
        recommendations.append(book_info)

    return recommendations

if __name__ == "__main__":
    print("\n📚 Welcome to the Book Recommendation System 📚")
    
    while True:
        user_query = input("\nDescribe the type of book you're looking for (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            print("\nThank you for using the Book Recommendation System! 📖✨")
            break

        results = recommend_books(user_query)
        print("\nTop Recommended Books:\n")
        for book in results:
            print(f"{book['Rank']}. {book['Title']} by {book['Author']}")
            print(f"   Description: {book['Description']}...\n")
