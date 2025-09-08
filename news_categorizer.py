# -*- coding: utf-8 -*-
"""
News Categorization System
Categorizes news articles into: business, politics, sports, entertainment, tech
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class NewsCategorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
        # Category keywords for rule-based classification
        self.category_keywords = {
            'business': [
                'business', 'economy', 'finance', 'stock', 'market', 'investment', 'banking',
                'corporate', 'earnings', 'revenue', 'profit', 'merger', 'acquisition', 'ipo',
                'startup', 'funding', 'venture', 'capital', 'trading', 'currency', 'inflation',
                'gdp', 'unemployment', 'federal reserve', 'interest rate', 'budget', 'tax'
            ],
            'technology': [
                'technology', 'tech', 'software', 'hardware', 'artificial intelligence', 'ai',
                'machine learning', 'ml', 'data science', 'cybersecurity', 'blockchain',
                'cryptocurrency', 'bitcoin', 'cloud computing', 'aws', 'azure', 'google cloud',
                'programming', 'coding', 'developer', 'app', 'mobile', 'smartphone', 'computer',
                'internet', 'digital', 'innovation', 'startup', 'silicon valley'
            ],
            'politics': [
                'politics', 'political', 'government', 'election', 'vote', 'candidate',
                'president', 'prime minister', 'minister', 'parliament', 'congress', 'senate',
                'policy', 'legislation', 'law', 'bill', 'democracy', 'republican', 'democrat',
                'campaign', 'poll', 'referendum', 'constitution', 'federal', 'state', 'local'
            ],
            'sports': [
                'sports', 'sport', 'football', 'soccer', 'basketball', 'baseball', 'tennis',
                'cricket', 'golf', 'hockey', 'olympics', 'world cup', 'championship', 'tournament',
                'player', 'team', 'coach', 'stadium', 'match', 'game', 'score', 'victory',
                'defeat', 'league', 'season', 'athlete', 'training', 'fitness'
            ],
            'entertainment': [
                'entertainment', 'movie', 'film', 'cinema', 'hollywood', 'bollywood',
                'music', 'song', 'album', 'concert', 'celebrity', 'actor', 'actress',
                'director', 'producer', 'television', 'tv', 'series', 'show', 'streaming',
                'netflix', 'amazon prime', 'disney', 'award', 'oscar', 'grammy', 'festival'
            ]
        }
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', '', text)
        # Tokenize and remove stopwords
        words = word_tokenize(text)
        words = [word for word in words if word not in stopwords.words('english')]
        return ' '.join(words)
    
    def rule_based_classification(self, text):
        """Rule-based classification using keywords"""
        text = str(text).lower()
        scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[category] = score
        
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return 'unclassified'
    
    def prepare_training_data(self, df):
        """Prepare training data from existing CSV"""
        # Clean text
        df['cleaned_text'] = df['content'].apply(self.clean_text)
        
        # Use existing categories or apply rule-based classification
        if 'category' in df.columns:
            df['predicted_category'] = df['category']
        else:
            df['predicted_category'] = df['content'].apply(self.rule_based_classification)
        
        return df
    
    def train_model(self, df):
        """Train the categorization model"""
        print("Preparing training data...")
        df = self.prepare_training_data(df)
        
        # Filter out unclassified articles
        df_classified = df[df['predicted_category'] != 'unclassified'].copy()
        
        if len(df_classified) == 0:
            print("No classified articles found for training!")
            return False
        
        print(f"Training with {len(df_classified)} classified articles")
        print("Category distribution:")
        print(df_classified['predicted_category'].value_counts())
        
        # Prepare features and labels
        X = self.vectorizer.fit_transform(df_classified['cleaned_text'])
        y = df_classified['predicted_category']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        return True
    
    def predict_categories(self, df):
        """Predict categories for new articles"""
        if not self.is_trained:
            print("Model not trained yet!")
            return df
        
        # Clean text
        df['cleaned_text'] = df['content'].apply(self.clean_text)
        
        # Predict categories
        X = self.vectorizer.transform(df['cleaned_text'])
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        df['predicted_category'] = predictions
        df['confidence'] = np.max(probabilities, axis=1)
        
        return df
    
    def save_model(self, filename='news_categorizer_model.joblib'):
        """Save the trained model"""
        if self.is_trained:
            model_data = {
                'vectorizer': self.vectorizer,
                'model': self.model,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, filename)
            print(f"Model saved to {filename}")
        else:
            print("No trained model to save!")
    
    def load_model(self, filename='news_categorizer_model.joblib'):
        """Load a pre-trained model"""
        if os.path.exists(filename):
            model_data = joblib.load(filename)
            self.vectorizer = model_data['vectorizer']
            self.model = model_data['model']
            self.is_trained = model_data['is_trained']
            print(f"Model loaded from {filename}")
            return True
        else:
            print(f"Model file {filename} not found!")
            return False

def main():
    """Main function to demonstrate the categorizer"""
    categorizer = NewsCategorizer()
    
    # Check if we have existing data to train on
    if os.path.exists('business_data.csv'):
        print("Loading existing data for training...")
        df = pd.read_csv('business_data.csv')
        
        # Train the model
        if categorizer.train_model(df):
            # Save the model
            categorizer.save_model()
            
            # Test on the same data
            df_with_predictions = categorizer.predict_categories(df)
            
            # Show results
            print("\nPrediction Results:")
            print(df_with_predictions[['headlines', 'predicted_category', 'confidence']].head(10))
            
            # Save results
            df_with_predictions.to_csv('categorized_news.csv', index=False)
            print("\nCategorized news saved to 'categorized_news.csv'")
        else:
            print("Training failed!")
    else:
        print("No existing data found. Please run news_fetcher.py first or provide a CSV file with 'content' column.")

if __name__ == "__main__":
    main()
