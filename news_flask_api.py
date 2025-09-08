from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import os
from final_mkt_analyzer import HybridEnricher, enrich_dataset

app = Flask(__name__)

model = None
enricher = HybridEnricher()

def load_model():
    global model
    try:
        model = joblib.load('news_multioutput_model.joblib')
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load model on startup
load_model()

def retrain_model():
    try:
        # Load Data
        CSV_PATH = 'business_tech_trainset.csv'
        if not os.path.exists(CSV_PATH):
            # Create training data from business_data.csv
            df = pd.read_csv('business_data.csv')
            df_enriched = enrich_dataset(df)
            df_enriched.to_csv(CSV_PATH, index=False)
        
        df = pd.read_csv(CSV_PATH)

        # Features and labels
        df = df.dropna(subset=['content', 'sector', 'reason', 'impact'])
        X = df['content']
        y = df[['sector', 'reason', 'impact']]

        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

        # Build model pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=2000, stop_words='english')),
            ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000)))
        ])

        # Train
        pipeline.fit(X_train, y_train)

        # Save model
        joblib.dump(pipeline, 'news_multioutput_model.joblib')

        # Evaluation
        score = pipeline.score(X_test, y_test)
        return score
    except Exception as e:
        raise e

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        score = retrain_model()
        # Reload the model
        if load_model():
            return jsonify({'message': 'Model retrained and reloaded successfully', 'test_score': score})
        else:
            return jsonify({'error': 'Model retrained but failed to reload'}), 500
    except Exception as e:
        return jsonify({'error': f'Retraining failed: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        texts = data.get('texts', [])
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'POST body must have a "texts": [ ... ] field'}), 400
        
        results = []
        for text in texts:
            # Use rule-based analysis if model not available
            if model is None:
                result = enricher.enrich(text)
                results.append({
                    'text': text,
                    'sector': result['sector'],
                    'reason': result['reason'],
                    'impact': result['impact']
                })
            else:
                # Use ML model
                preds = model.predict([text])
                results.append({
                    'text': text,
                    'sector': preds[0][0],
                    'reason': preds[0][1],
                    'impact': preds[0][2]
                })
        
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'News Analysis API',
        'endpoints': {
            '/predict': 'POST - Analyze news text',
            '/retrain': 'POST - Retrain model',
            '/health': 'GET - Health check'
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)