# -*- coding: utf-8 -*-
"""
Market Trend Analyzer - News Impact Analysis
A comprehensive tool for analyzing market impact of news articles
"""

import pandas as pd
import numpy as np
import os
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Optional imports for advanced features
try:
    from matplotlib import pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Download NLTK data safely
def download_nltk_data():
    """Download required NLTK data if not already present"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

# Initialize NLTK data
download_nltk_data()

class HybridEnricher:
    def __init__(self, use_rules=True):
        self.use_rules = use_rules

    # -------------------------------
    # Rule-based enrichment (12 macro rules)
    # -------------------------------
    def apply_rules(self, content: str):
        text = content.lower()

        # Case 1: Raw material prices
        if "raw material" in text or "commodity price" in text:
            if "increase" in text or "rise" in text:
                return {"sector": "Manufacturing", "reason": "Raw material cost up -> profit margin down", "impact": "Negative"}
            if "decrease" in text or "fall" in text:
                return {"sector": "Manufacturing", "reason": "Raw material cost down -> profit margin up", "impact": "Positive"}

        # Case 2: Production capacity
        if "increase production" in text or "expand capacity" in text or "new plant" in text or "induct aircraft" in text:
            return {"sector": "Manufacturing" if "plant" in text else "Aviation",
                    "reason": "Capacity expansion -> higher sales", "impact": "Positive"}
        if "production issue" in text or "pipeline issue" in text or "halt production" in text:
            return {"sector": "Manufacturing", "reason": "Production disruption -> reduced sales", "impact": "Negative"}

        # Case 3: Taxes / subsidies
        if "tax" in text or "subsidy" in text:
            if "reduce" in text or "cut" in text or "relax" in text:
                return {"sector": "Macro Economy", "reason": "Tax relief / subsidy -> higher consumption", "impact": "Positive"}
            if "increase" in text or "remove" in text or "hike" in text:
                return {"sector": "Macro Economy", "reason": "Higher taxes / subsidy removal -> lower consumption", "impact": "Negative"}

        # Case 4: Disposable income (income tax)
        if "income tax" in text:
            if "reduce" in text or "cut" in text:
                return {"sector": "Macro Economy", "reason": "Lower income tax -> more disposable income", "impact": "Positive"}
            if "increase" in text or "hike" in text:
                return {"sector": "Macro Economy", "reason": "Higher income tax -> less disposable income", "impact": "Negative"}

        # Case 5: Tariffs, sanctions, trade
        if "tariff" in text or "sanction" in text or "trade deal" in text or "export" in text:
            if "impose" in text or "increase" in text or "halt" in text:
                return {"sector": "Macro Economy", "reason": "Tariffs / sanctions -> reduced exports", "impact": "Negative"}
            if "reduce" in text or "revoke" in text or "agreement" in text:
                return {"sector": "Macro Economy", "reason": "Tariff relief / trade deals -> increased exports", "impact": "Positive"}

        # Case 6: Interest rates
        if "interest rate" in text or "repo rate" in text:
            if "reduce" in text or "cut" in text or "lower" in text:
                return {"sector": "Macro Economy", "reason": "Lower interest rate -> more loans & investment", "impact": "Positive"}
            if "increase" in text or "hike" in text:
                return {"sector": "Macro Economy", "reason": "Higher interest rate -> borrowing cost up -> spending down", "impact": "Negative"}

        # Case 7: Inflation
        if "inflation" in text:
            if "increase" in text or "high" in text:
                return {"sector": "Macro Economy", "reason": "High inflation -> reduced consumption", "impact": "Negative"}
            if "cool" in text or "decrease" in text or "fall" in text:
                return {"sector": "Macro Economy", "reason": "Lower inflation -> higher consumption", "impact": "Positive"}

        # Case 8: Currency strength
        if "rupee" in text or "currency" in text or "exchange rate" in text:
            if "strengthen" in text or "appreciate" in text:
                return {"sector": "Export-driven", "reason": "Stronger currency -> exports less competitive", "impact": "Negative"}
            if "weaken" in text or "depreciate" in text:
                return {"sector": "Export-driven", "reason": "Weaker currency -> exports more competitive", "impact": "Positive"}

        # Case 9: Fraud
        if "fraud" in text or "scam" in text or "embezzlement" in text:
            return {"sector": "Company-specific", "reason": "Corporate fraud -> loss of trust", "impact": "Negative"}

        # Case 10: New technology
        if "new technology" in text or "innovation" in text or "ai" in text or "automation" in text:
            return {"sector": "Technology", "reason": "New tech -> efficiency up", "impact": "Positive"}

        # Case 11: Government capex
        if "government spending" in text or "capex" in text or "infrastructure project" in text:
            return {"sector": "Infrastructure", "reason": "Govt capex -> demand up", "impact": "Positive"}

        # Case 12: Employment
        if "unemployment" in text or "jobs" in text:
            if "increase" in text or "rise" in text:
                return {"sector": "Macro Economy", "reason": "Unemployment up -> weaker consumption", "impact": "Negative"}
            if "decrease" in text or "fall" in text or "drop" in text:
                return {"sector": "Macro Economy", "reason": "Unemployment down -> stronger consumption", "impact": "Positive"}

        return None

    # -------------------------------
    # Sentiment fallback
    # -------------------------------
    def sentiment_enrich(self, content: str):
        polarity = TextBlob(content).sentiment.polarity
        impact = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Mixed"
        return {"sector": "General", "reason": "Sentiment-based fallback", "impact": impact}

    # -------------------------------
    # Normalization
    # -------------------------------
    def normalize_impact(self, value: str):
        value = str(value).lower().strip()
        if "positive" in value: return "Positive"
        if "negative" in value: return "Negative"
        if "mixed" in value: return "Mixed"
        return "Mixed"

    # -------------------------------
    # Main enrichment
    # -------------------------------
    def enrich(self, content: str):
        result = None
        if self.use_rules:
            result = self.apply_rules(content)
        if not result:
            result = self.sentiment_enrich(content)
        result["impact"] = self.normalize_impact(result.get("impact", "Mixed"))
        return result

# ===========================
# Utility Functions
# ===========================

def enrich_dataset(df: pd.DataFrame, enricher=None):
    """
    Enrich entire dataset with market impact analysis
    
    Args:
        df: DataFrame with 'content' column
        enricher: HybridEnricher instance (optional)
    
    Returns:
        DataFrame with enriched data
    """
    if enricher is None:
        enricher = HybridEnricher()
    
    enriched_data = []
    for _, row in df.iterrows():
        content = str(row.get("content", ""))
        enriched = enricher.enrich(content)
        
        # Preserve original columns and add enrichment
        row_data = row.to_dict()
        row_data.update({
            "sector": enriched["sector"],
            "reason": enriched["reason"],
            "impact": enriched["impact"]
        })
        enriched_data.append(row_data)

    return pd.DataFrame(enriched_data)

def analyze_impact_distribution(df: pd.DataFrame, plot=False):
    """
    Analyze and optionally plot impact distribution
    
    Args:
        df: DataFrame with 'impact' column
        plot: Whether to create a plot (requires matplotlib)
    
    Returns:
        Series with impact counts
    """
    if 'impact' not in df.columns:
        raise ValueError("DataFrame must contain 'impact' column")
    
    impact_counts = df["impact"].value_counts()
    print("Impact Distribution:")
    print(impact_counts)
    
    if plot and MATPLOTLIB_AVAILABLE:
        try:
            impact_counts.plot(kind="bar", figsize=(6,4), 
                             title="Impact Count (Positive / Negative / Mixed)")
            plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")
    elif plot:
        print("Matplotlib not available for plotting")
    
    return impact_counts

def clean_text(text):
    """
    Clean text for ML processing
    
    Args:
        text: Input text string
    
    Returns:
        Cleaned text string
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    
    try:
        words = word_tokenize(text)
        words = [word for word in words if word not in stopwords.words('english')]
        return ' '.join(words)
    except:
        # Fallback if NLTK fails
        return ' '.join(text.split())

def train_impact_classifier(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Train a classifier for impact prediction
    
    Args:
        df: DataFrame with 'content' and 'impact' columns
        test_size: Fraction of data for testing
        random_state: Random seed
    
    Returns:
        Tuple of (model, vectorizer, accuracy, report)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for ML training")
    
    if 'content' not in df.columns or 'impact' not in df.columns:
        raise ValueError("DataFrame must contain 'content' and 'impact' columns")
    
    # Prepare data
    df['cleaned_content'] = df['content'].apply(clean_text)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['cleaned_content'])
    y = df['impact']
    
    # Handle class imbalance if SMOTE is available
    try:
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    except:
        X_resampled, y_resampled = X, y
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=test_size, random_state=random_state
    )
    
    # Train model
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    try:
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred)
    except:
        report = f"Accuracy: {accuracy:.4f}"
    
    return model, vectorizer, accuracy, report

# ===========================
# Main execution (only runs if script is executed directly)
# ===========================

def main():
    """
    Main function to demonstrate the analyzer
    """
    print("Market Trend Analyzer - News Impact Analysis")
    print("=" * 50)
    
    # Check if business_data.csv exists
    if os.path.exists("business_data.csv"):
        print("Loading business_data.csv...")
        df = pd.read_csv("business_data.csv")
        
        # Enrich dataset
        print("Enriching dataset...")
        df_enriched = enrich_dataset(df)
        
        # Analyze distribution
        analyze_impact_distribution(df_enriched, plot=MATPLOTLIB_AVAILABLE)
        
        # Save enriched data
        output_file = "enriched_business_data.csv"
        df_enriched.to_csv(output_file, index=False)
        print(f"\nEnriched data saved to: {output_file}")
        
        # Train classifier if sklearn is available
        if SKLEARN_AVAILABLE and len(df_enriched) > 10:
            print("\nTraining impact classifier...")
            try:
                model, vectorizer, accuracy, report = train_impact_classifier(df_enriched)
                print(f"Model accuracy: {accuracy:.4f}")
                print("\nClassification Report:")
                print(report)
            except Exception as e:
                print(f"Training failed: {e}")
        
        return df_enriched
    
    else:
        print("business_data.csv not found. Testing with sample data...")
        
        # Create sample data
        sample_data = [
            "Interest rates are rising, affecting borrowing costs for businesses.",
            "New AI technology promises to revolutionize manufacturing processes.",
            "Government announces tax cuts to stimulate economic growth.",
            "Raw material prices surge due to supply chain disruptions."
        ]
        
        enricher = HybridEnricher()
        
        print("\nSample Analysis:")
        for i, text in enumerate(sample_data, 1):
            result = enricher.enrich(text)
            print(f"\n{i}. Text: {text[:60]}...")
            print(f"   Sector: {result['sector']}")
            print(f"   Impact: {result['impact']}")
            print(f"   Reason: {result['reason']}")
        
        return None

if __name__ == "__main__":
    main()