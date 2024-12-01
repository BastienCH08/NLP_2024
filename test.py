# Required imports
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from rank_bm25 import BM25Okapi

# Load sampled data
def load_sampled_data(file_path):
    print(f"\nLoading cleaned data from {file_path}...")
    data = pd.read_csv(file_path)
    print(f"Successfully loaded {len(data)} reviews")
    return data.sample(n=100, random_state=42)  # Sample 100 rows for testing

# Generate embeddings using SentenceTransformer
def generate_embeddings(data, batch_size=16):
    print("\nStarting embedding generation...")
    model = SentenceTransformer('all-mpnet-base-v2', device='cpu')  # Use CPU
    embeddings = []

    for i in tqdm(range(0, len(data), batch_size), desc="Generating embeddings"):
        batch = data['cleaned_text'].iloc[i:i + batch_size].tolist()
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)

    data['embedding'] = embeddings
    print("Embedding generation complete.")
    return data

# Compute semantic similarity
def compute_similarity(data, query):
    print(f"\nProcessing query: '{query}'")
    model = SentenceTransformer('all-mpnet-base-v2', device='cpu')  # Use CPU
    query_embedding = model.encode(query, show_progress_bar=False)
    candidate_embeddings = np.array(list(data['embedding']))
    similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
    data['semantic_similarity'] = similarities
    print("Similarity computation completed.")
    return data

# Prepare data for BM25
def prepare_for_bm25(data):
    print("\nPreparing data for BM25...")
    data = data.dropna(subset=['cleaned_text'])  # Drop rows with missing cleaned_text
    data['cleaned_text'] = data['cleaned_text'].astype(str)  # Ensure all texts are strings
    print(f"BM25 preparation complete. Total valid rows: {len(data)}")
    return data

# BM25 implementation
def run_bm25(data, query, top_k=5):
    print("\nRunning BM25...")
    tokenized_corpus = [doc.split() for doc in data['cleaned_text']]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    data['bm25_score'] = scores
    sorted_data = data.sort_values(by='bm25_score', ascending=False).head(top_k)
    print("BM25 computation completed.")
    return data, sorted_data

# Calculate Mean Squared Error (MSE)
def calculate_mse(model_scores, bm25_scores):
    scaler = MinMaxScaler()
    model_scores_normalized = scaler.fit_transform(np.array(model_scores).reshape(-1, 1)).flatten()
    bm25_scores_normalized = scaler.fit_transform(np.array(bm25_scores).reshape(-1, 1)).flatten()
    mse = mean_squared_error(model_scores_normalized, bm25_scores_normalized)
    return mse

# Main function
def main():
    try:
        # File path for cleaned reviews
        file_path = r"D:\ESILV_2024-2025\Crous\NLP\lab\NLP_2024\archive\cleaned_reviews.csv"

        # Load cleaned data and sample 100 rows
        sampled_data = load_sampled_data(file_path)

        # Generate embeddings
        sampled_data = generate_embeddings(sampled_data)

        # Query for similarity comparison
        query = "I enjoyed the cozy atmosphere and excellent service."

        # Compute semantic similarity
        sampled_data = compute_similarity(sampled_data, query)

        # Prepare data for BM25
        sampled_data = prepare_for_bm25(sampled_data)

        # Run BM25
        sampled_data, bm25_results = run_bm25(sampled_data, query)

        # Calculate MSE between semantic similarity and BM25 scores
        mse_value = calculate_mse(sampled_data['semantic_similarity'], sampled_data['bm25_score'])
        print(f"\nMSE between the semantic model and BM25: {mse_value}")

        # Display top 5 results for both models
        print("\nSemantic Model Top 5 Recommendations:")
        print(sampled_data.sort_values(by='semantic_similarity', ascending=False)[['cleaned_text', 'semantic_similarity']].head(5))
        
        print("\nBM25 Top 5 Recommendations:")
        print(bm25_results[['cleaned_text', 'bm25_score']])

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
