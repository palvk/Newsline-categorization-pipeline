# -*- coding: utf-8 -*-
"""
Market Impact Analyzer
Analyzes business and tech news to assign sector, impact, and reason tags
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class MarketImpactAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.impact_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
        # Enhanced rule-based enrichment with more comprehensive rules
        self.sector_keywords = {
            'Manufacturing': [
                'manufacturing', 'production', 'factory', 'plant', 'assembly', 'raw material',
                'commodity', 'supply chain', 'logistics', 'automotive', 'textile', 'steel',
                'aluminum', 'chemical', 'pharmaceutical', 'food processing', 'electronics'
            ],
            'Technology': [
                'technology', 'software', 'hardware', 'ai', 'artificial intelligence',
                'machine learning', 'data science', 'cybersecurity', 'blockchain',
                'cloud computing', 'saas', 'paas', 'iaas', 'startup', 'fintech',
                'edtech', 'healthtech', 'biotech', 'semiconductor', 'chip'
            ],
            'Financial Services': [
                'banking', 'finance', 'investment', 'insurance', 'mutual fund',
                'hedge fund', 'private equity', 'venture capital', 'trading',
                'stock market', 'bond', 'derivative', 'credit', 'loan', 'mortgage'
            ],
            'Energy': [
                'energy', 'oil', 'gas', 'petroleum', 'renewable', 'solar', 'wind',
                'nuclear', 'coal', 'electricity', 'power', 'utility', 'grid'
            ],
            'Healthcare': [
                'healthcare', 'medical', 'pharmaceutical', 'drug', 'vaccine',
                'hospital', 'clinic', 'biotech', 'medical device', 'diagnostic'
            ],
            'Infrastructure': [
                'infrastructure', 'construction', 'real estate', 'transportation',
                'railway', 'airport', 'highway', 'bridge', 'tunnel', 'urban development'
            ],
            'Retail': [
                'retail', 'e-commerce', 'shopping', 'consumer', 'brand', 'fashion',
                'grocery', 'supermarket', 'mall', 'online shopping', 'delivery'
            ],
            'Aviation': [
                'aviation', 'airline', 'aircraft', 'airport', 'flight', 'pilot',
                'aircraft', 'aerospace', 'defense', 'military'
            ],
            'Macro Economy': [
                'economy', 'gdp', 'inflation', 'unemployment', 'interest rate',
                'federal reserve', 'central bank', 'monetary policy', 'fiscal policy',
                'tax', 'budget', 'deficit', 'surplus', 'trade', 'export', 'import'
            ]
        }
        
        self.impact_rules = {
            'positive_keywords': [
                'increase', 'rise', 'growth', 'expansion', 'boost', 'surge', 'jump',
                'profit', 'revenue', 'earnings', 'success', 'breakthrough', 'innovation',
                'partnership', 'deal', 'agreement', 'investment', 'funding', 'ipo',
                'merger', 'acquisition', 'launch', 'release', 'upgrade', 'improvement'
            ],
            'negative_keywords': [
                'decrease', 'fall', 'decline', 'drop', 'loss', 'recession', 'crisis',
                'bankruptcy', 'layoff', 'cut', 'reduction', 'shortage', 'delay',
                'failure', 'scandal', 'fraud', 'investigation', 'fine', 'penalty',
                'lawsuit', 'dispute', 'conflict', 'war', 'sanction', 'tariff'
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
    
    def determine_sector(self, text):
        """Determine sector based on content"""
        text = str(text).lower()
        sector_scores = {}
        
        for sector, keywords in self.sector_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            sector_scores[sector] = score
        
        if max(sector_scores.values()) > 0:
            return max(sector_scores, key=sector_scores.get)
        return 'General'
    
    def determine_impact(self, text):
        """Determine market impact using enhanced rule-based approach"""
        text = str(text).lower()
        
        # Enhanced keyword matching with weights
        positive_score = 0
        negative_score = 0
        
        # Strong positive indicators (weight 2)
        strong_positive = ['breakthrough', 'record profit', 'surge', 'boom', 'rally', 'soar', 'skyrocket', 'outperform']
        for keyword in strong_positive:
            if keyword in text:
                positive_score += 2
        
        # Regular positive indicators (weight 1)
        for keyword in self.impact_rules['positive_keywords']:
            if keyword in text:
                positive_score += 1
        
        # Strong negative indicators (weight 2)
        strong_negative = ['crash', 'collapse', 'bankruptcy', 'crisis', 'recession', 'plunge', 'tumble', 'slump']
        for keyword in strong_negative:
            if keyword in text:
                negative_score += 2
        
        # Regular negative indicators (weight 1)
        for keyword in self.impact_rules['negative_keywords']:
            if keyword in text:
                negative_score += 1
        
        # Context-aware analysis
        # Check for financial numbers and their context
        import re
        numbers = re.findall(r'\$[\d,]+(?:\.\d+)?[kmb]?|\d+(?:\.\d+)?%', text)
        if numbers:
            # If there are financial numbers, weight the sentiment more heavily
            sentiment_multiplier = 1.5
        else:
            sentiment_multiplier = 1.0
        
        # Enhanced sentiment analysis
        sentiment = TextBlob(text).sentiment
        sentiment_score = sentiment.polarity * sentiment_multiplier
        
        # Combine rule-based and sentiment scores
        total_positive = positive_score + (sentiment_score if sentiment_score > 0 else 0)
        total_negative = negative_score + (abs(sentiment_score) if sentiment_score < 0 else 0)
        
        # Determine impact with threshold
        if total_positive > total_negative and total_positive > 1:
            return 'Positive'
        elif total_negative > total_positive and total_negative > 1:
            return 'Negative'
        else:
            return 'Mixed'
    
    def determine_reason(self, text, sector, impact):
        """Determine reason for impact based on enhanced content analysis"""
        text = str(text).lower()
        
        # Enhanced reason patterns with more specific detection
        reasons = []
        
        # Financial indicators
        if 'raw material' in text or 'commodity' in text or 'supply chain' in text:
            if impact == 'Positive':
                reasons.append('Raw material cost ↓ → profit margin ↑')
            else:
                reasons.append('Raw material cost ↑ → profit margin ↓')
        
        if 'production' in text or 'capacity' in text or 'manufacturing' in text:
            if impact == 'Positive':
                reasons.append('Capacity expansion → higher sales')
            else:
                reasons.append('Production disruption → reduced sales')
        
        if 'tax' in text or 'subsidy' in text or 'government policy' in text:
            if impact == 'Positive':
                reasons.append('Tax relief / subsidy → higher consumption')
            else:
                reasons.append('Higher taxes / subsidy removal → lower consumption')
        
        if 'interest rate' in text or 'repo rate' in text or 'federal reserve' in text:
            if impact == 'Positive':
                reasons.append('Lower interest rate → more loans & investment')
            else:
                reasons.append('Higher interest rate → borrowing cost ↑ → spending ↓')
        
        if 'inflation' in text or 'cpi' in text or 'price level' in text:
            if impact == 'Positive':
                reasons.append('Lower inflation → higher consumption')
            else:
                reasons.append('High inflation → reduced consumption')
        
        if 'currency' in text or 'rupee' in text or 'dollar' in text or 'exchange rate' in text:
            if impact == 'Positive':
                reasons.append('Weaker currency → exports more competitive')
            else:
                reasons.append('Stronger currency → exports less competitive')
        
        # Corporate events
        if 'fraud' in text or 'scam' in text or 'scandal' in text:
            reasons.append('Corporate fraud → loss of trust')
        
        if 'merger' in text or 'acquisition' in text or 'deal' in text:
            if impact == 'Positive':
                reasons.append('Strategic merger/acquisition → market consolidation')
            else:
                reasons.append('Failed deal → market uncertainty')
        
        # Technology and innovation
        if 'technology' in text or 'innovation' in text or 'ai' in text or 'artificial intelligence' in text:
            reasons.append('New tech → efficiency ↑')
        
        if 'startup' in text or 'funding' in text or 'investment' in text:
            if impact == 'Positive':
                reasons.append('Increased funding → innovation boost')
            else:
                reasons.append('Funding shortage → innovation slowdown')
        
        # Government and policy
        if 'government spending' in text or 'capex' in text or 'infrastructure' in text:
            reasons.append('Govt capex → demand ↑')
        
        if 'regulation' in text or 'policy' in text or 'compliance' in text:
            if impact == 'Positive':
                reasons.append('Favorable regulation → business growth')
            else:
                reasons.append('Strict regulation → compliance costs ↑')
        
        # Employment and labor
        if 'unemployment' in text or 'jobs' in text or 'hiring' in text:
            if impact == 'Positive':
                reasons.append('Unemployment ↓ → stronger consumption')
            else:
                reasons.append('Unemployment ↑ → weaker consumption')
        
        # Market sentiment indicators
        if 'bull market' in text or 'rally' in text or 'surge' in text:
            reasons.append('Market rally → investor confidence ↑')
        
        if 'bear market' in text or 'crash' in text or 'decline' in text:
            reasons.append('Market decline → investor confidence ↓')
        
        # Sector-specific reasons
        if sector == 'Technology':
            if 'software' in text or 'cloud' in text:
                reasons.append('Tech innovation → digital transformation')
            elif 'cybersecurity' in text or 'security' in text:
                reasons.append('Security concerns → tech investment ↑')
        
        elif sector == 'Financial Services':
            if 'banking' in text or 'credit' in text:
                reasons.append('Banking sector performance → economic health')
            elif 'crypto' in text or 'bitcoin' in text:
                reasons.append('Crypto market volatility → financial uncertainty')
        
        elif sector == 'Energy':
            if 'oil' in text or 'gas' in text:
                reasons.append('Energy prices → economic impact')
            elif 'renewable' in text or 'solar' in text or 'wind' in text:
                reasons.append('Renewable energy → sustainable growth')
        
        # Return the most relevant reason or combine multiple reasons
        if reasons:
            if len(reasons) == 1:
                return reasons[0]
            else:
                return f"{reasons[0]} | {reasons[1]}" if len(reasons) > 1 else reasons[0]
        else:
            # Generic reason based on sentiment and impact
            sentiment = TextBlob(text).sentiment.polarity
            if impact == 'Positive':
                if sentiment > 0.2:
                    return 'Strong positive sentiment → market optimism'
                else:
                    return 'Positive market sentiment → increased confidence'
            elif impact == 'Negative':
                if sentiment < -0.2:
                    return 'Strong negative sentiment → market pessimism'
                else:
                    return 'Negative market sentiment → reduced confidence'
            else:
                return 'Mixed signals → uncertain market outlook'
    
    def analyze_articles(self, df):
        """Analyze articles and assign sector, impact, and reason"""
        print("Analyzing articles for market impact...")
        
        # Filter for business and tech articles only
        business_tech_df = df[df['predicted_category'].isin(['business', 'technology'])].copy()
        
        if len(business_tech_df) == 0:
            print("No business or technology articles found!")
            return df
        
        print(f"Analyzing {len(business_tech_df)} business/tech articles...")
        
        # Apply analysis
        business_tech_df['sector'] = business_tech_df['content'].apply(self.determine_sector)
        business_tech_df['impact'] = business_tech_df['content'].apply(self.determine_impact)
        business_tech_df['reason'] = business_tech_df.apply(
            lambda row: self.determine_reason(row['content'], row['sector'], row['impact']), 
            axis=1
        )
        
        # Update the original dataframe
        df_updated = df.copy()
        for idx, row in business_tech_df.iterrows():
            df_updated.loc[idx, 'sector'] = row['sector']
            df_updated.loc[idx, 'impact'] = row['impact']
            df_updated.loc[idx, 'reason'] = row['reason']
        
        return df_updated
    
    def train_impact_model(self, df):
        """Train a machine learning model for impact prediction"""
        # Filter for business/tech articles with impact labels
        training_df = df[
            (df['predicted_category'].isin(['business', 'technology'])) & 
            (df['impact'].notna())
        ].copy()
        
        if len(training_df) == 0:
            print("No training data available for impact model!")
            return False
        
        print(f"Training impact model with {len(training_df)} articles...")
        
        # Prepare features
        training_df['cleaned_text'] = training_df['content'].apply(self.clean_text)
        X = self.vectorizer.fit_transform(training_df['cleaned_text'])
        y = training_df['impact']
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )
        
        # Train model
        self.impact_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.impact_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Impact Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        return True
    
    def predict_impact_ml(self, df):
        """Predict impact using trained ML model"""
        if not self.is_trained:
            print("Impact model not trained yet!")
            return df
        
        # Filter for business/tech articles
        business_tech_df = df[df['predicted_category'].isin(['business', 'technology'])].copy()
        
        if len(business_tech_df) == 0:
            return df
        
        # Prepare features
        business_tech_df['cleaned_text'] = business_tech_df['content'].apply(self.clean_text)
        X = self.vectorizer.transform(business_tech_df['cleaned_text'])
        
        # Predict
        predictions = self.impact_model.predict(X)
        probabilities = self.impact_model.predict_proba(X)
        
        business_tech_df['ml_impact'] = predictions
        business_tech_df['ml_confidence'] = np.max(probabilities, axis=1)
        
        # Update original dataframe
        df_updated = df.copy()
        for idx, row in business_tech_df.iterrows():
            df_updated.loc[idx, 'ml_impact'] = row['ml_impact']
            df_updated.loc[idx, 'ml_confidence'] = row['ml_confidence']
        
        return df_updated
    
    def save_model(self, filename='market_impact_model.joblib'):
        """Save the trained model"""
        if self.is_trained:
            model_data = {
                'vectorizer': self.vectorizer,
                'impact_model': self.impact_model,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, filename)
            print(f"Market impact model saved to {filename}")
        else:
            print("No trained model to save!")
    
    def load_model(self, filename='market_impact_model.joblib'):
        """Load a pre-trained model"""
        if os.path.exists(filename):
            model_data = joblib.load(filename)
            self.vectorizer = model_data['vectorizer']
            self.impact_model = model_data['impact_model']
            self.is_trained = model_data['is_trained']
            print(f"Market impact model loaded from {filename}")
            return True
        else:
            print(f"Model file {filename} not found!")
            return False

def main():
    """Main function to demonstrate the analyzer"""
    analyzer = MarketImpactAnalyzer()
    
    # Check if we have categorized news
    if os.path.exists('categorized_news.csv'):
        print("Loading categorized news...")
        df = pd.read_csv('categorized_news.csv')
        
        # Analyze articles
        df_analyzed = analyzer.analyze_articles(df)
        
        # Train ML model if we have enough data
        if len(df_analyzed[df_analyzed['predicted_category'].isin(['business', 'technology'])]) > 50:
            analyzer.train_impact_model(df_analyzed)
            analyzer.save_model()
        
        # Save results
        df_analyzed.to_csv('business_tech_trainset.csv', index=False)
        print("\nBusiness/Tech training set saved to 'business_tech_trainset.csv'")
        
        # Show results
        business_tech = df_analyzed[df_analyzed['predicted_category'].isin(['business', 'technology'])]
        if len(business_tech) > 0:
            print(f"\nAnalyzed {len(business_tech)} business/tech articles")
            print("\nSector Distribution:")
            print(business_tech['sector'].value_counts())
            print("\nImpact Distribution:")
            print(business_tech['impact'].value_counts())
            
            print("\nSample Results:")
            print(business_tech[['headlines', 'sector', 'impact', 'reason']].head())
    else:
        print("No categorized news found. Please run news_categorizer.py first.")

if __name__ == "__main__":
    main()
