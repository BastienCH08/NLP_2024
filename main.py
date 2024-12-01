import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import warnings
import json

# Ignore warnings
warnings.filterwarnings('ignore')

# Ensure all necessary NLTK data is downloaded
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        print("NLTK data download complete")
    except Exception as e:
        print(f"NLTK data download failed: {e}")

# Download NLTK data
download_nltk_data()

class HotelRecommender:
    def __init__(self, base_path, batch_size=32):
        """Initialize the recommender"""
        self.base_path = base_path
        self.batch_size = batch_size
        self.model = SentenceTransformer('all-mpnet-base-v2')  # Load model
        self.reviews = None

    def load_data(self):
        """Load data"""
        try:
            reviews_path = os.path.join(self.base_path, "reviews.csv")
            self.reviews = pd.read_csv(reviews_path)
            print(f"Successfully loaded {len(self.reviews)} reviews")
            
            # Parse ratings string to dictionary
            self.reviews['ratings'] = self.reviews['ratings'].apply(lambda x: json.loads(x.replace("'", '"')))
            return True
        except Exception as e:
            print(f"Data loading failed: {e}")
            return False

    @staticmethod
    def clean_text(text):
        """Text cleaning"""
        try:
            # Basic cleaning
            text = re.sub(r'<.*?>', '', str(text))  # Remove HTML tags
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Remove non-alphabetic characters
            text = text.lower()  # Convert to lowercase
            
            # Tokenization and stopword removal
            words = text.split()  # Use simple tokenization
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words]
            
            return ' '.join(words)
        except Exception as e:
            print(f"Text cleaning failed: {e}")
            return ""

    def preprocess_data(self):
        """Data preprocessing"""
        try:
            print("Starting data preprocessing...")
            cleaned_texts = []
            
            # Use tqdm to show progress
            for text in tqdm(self.reviews['text'], desc="Cleaning text"):
                cleaned_texts.append(self.clean_text(text))
            
            self.reviews['cleaned_text'] = cleaned_texts
            
            # Save preprocessed data
            cleaned_path = os.path.join(self.base_path, "cleaned_reviews.csv")
            self.reviews.to_csv(cleaned_path, index=False)
            print("Data preprocessing complete")
            return True
        except Exception as e:
            print(f"Data preprocessing failed: {e}")
            return False

    def generate_embeddings(self):
        """Generate text embeddings"""
        try:
            print("Starting embedding generation...")
            embeddings = []

            # Batch processing to generate embeddings
            for i in tqdm(range(0, len(self.reviews), self.batch_size)):
                batch = self.reviews['cleaned_text'].iloc[i:i + self.batch_size].tolist()
                batch_embeddings = self.model.encode(batch, show_progress_bar=False)
                embeddings.extend(batch_embeddings)

            self.reviews['embedding'] = embeddings
            
            # Save embeddings
            embeddings_path = os.path.join(self.base_path, "reviews_with_embeddings.pkl")
            self.reviews.to_pickle(embeddings_path)
            print("Embedding generation complete")
            return True
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            return False

    def compute_similarity(self, query_text):
        """Compute similarity"""
        try:
            # Clean query text
            cleaned_query = self.clean_text(query_text)
            query_embedding = self.model.encode(cleaned_query, show_progress_bar=False)
            
            # Get embeddings of all reviews
            candidate_embeddings = np.array(list(self.reviews['embedding']))
            
            # Compute cosine similarity
            similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
            self.reviews['semantic_similarity'] = similarities
            return True
        except Exception as e:
            print(f"Similarity computation failed: {e}")
            return False

    def calculate_final_scores(self, weights):
        """Calculate final scores"""
        try:
            alpha, beta, gamma, delta = weights
            self.reviews['final_score'] = alpha * self.reviews['semantic_similarity']
            
            # Sort and save
            results_path = os.path.join(self.base_path, "final_results.csv")
            self.reviews.sort_values(by='final_score', ascending=False).to_csv(results_path, index=False)
            print("Final score calculation complete")
            return True
        except Exception as e:
            print(f"Final score calculation failed: {e}")
            return False

def main():
    base_path = r"D:\ESILV_2024-2025\Crous\NLP\lab\NLP_2024\archive"
    recommender = HotelRecommender(base_path)

    # Execution flow
    if recommender.load_data():
        if recommender.preprocess_data():
            if recommender.generate_embeddings():
                query = "I loved the clean environment and friendly staff."
                if recommender.compute_similarity(query):
                    weights = (0.4, 0.3, 0.2, 0.1)
                    recommender.calculate_final_scores(weights)

if __name__ == "__main__":
    main()