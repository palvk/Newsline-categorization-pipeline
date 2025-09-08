# -*- coding: utf-8 -*-
"""
Simplified Streamlit News Analysis Dashboard
Focused on text input and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import joblib
import io
import base64
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Import our custom modules
try:
    from news_fetcher import NewsFetcher
    from news_categorizer import NewsCategorizer
    from market_impact_analyzer import MarketImpactAnalyzer
    from visualizations import NewsVisualizer
except ImportError:
    # Fallback if modules not available
    NewsFetcher = None
    NewsCategorizer = None
    MarketImpactAnalyzer = None
    NewsVisualizer = None

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Page configuration
st.set_page_config(
    page_title="News Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #2E86AB, #A23B72);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2C3E50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2E86AB;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'enriched_data' not in st.session_state:
    st.session_state.enriched_data = None

# Initialize components
@st.cache_resource
def load_components():
    """Load and cache analysis components"""
    try:
        if NewsFetcher:
            fetcher = NewsFetcher("17bb213791da4effb5e2ac8f0d3ef504")
        else:
            fetcher = None
        
        return None, None, None, fetcher
    except Exception as e:
        st.error(f"Error loading components: {e}")
        return None, None, None, None

categorizer, analyzer, visualizer, fetcher = load_components()

# Main header
st.markdown('<h1 class="main-header">📊 News Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6C757D;">Real-time Market Impact Analysis & Machine Learning Pipeline</p>', unsafe_allow_html=True)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Text Analysis", 
    "🤖 Model Training & Metrics", 
    "📈 Visualizations", 
    "📚 Project Documentation"
])

# Tab 1: Text Analysis
with tab1:
    st.markdown('<h2 class="section-header">🔍 Text Analysis</h2>', unsafe_allow_html=True)
    
    # Text input section
    st.subheader("📝 Enter News Article")
    
    # Text input options
    input_method = st.radio("Choose input method:", ["Paste Text", "Fetch from NewsAPI"])
    
    if input_method == "Paste Text":
        article_text = st.text_area(
            "Paste your news article here:",
            height=300,
            placeholder="Enter the complete news article text here...",
            help="Paste the entire news article content for analysis"
        )
        
        if st.button("🔍 Analyze Article", type="primary", use_container_width=True):
            if article_text.strip():
                with st.spinner("Analyzing article..."):
                    try:
                        # Create article dataframe
                        article_df = pd.DataFrame([{
                            'headlines': 'Custom Article',
                            'content': article_text,
                            'url': '',
                            'source': 'User Input'
                        }])
                        
                        # Use final_mkt_analyzer for analysis
                        from final_mkt_analyzer import enrich_dataset
                        article_df = enrich_dataset(article_df)
                        
                        # Add predicted_category if missing
                        if 'predicted_category' not in article_df.columns:
                            article_df['predicted_category'] = 'business'
                        
                        # Store for visualization
                        st.session_state.enriched_data = article_df
                        
                        # Display results
                        st.subheader("📊 Analysis Results")
                        
                        result = article_df.iloc[0]
                        
                        # Main results in columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📋 Article Information:**")
                            st.info(f"**Category:** {result['predicted_category']}")
                            st.info(f"**Sector:** {result.get('sector', 'N/A')}")
                        
                        with col2:
                            st.markdown("**📈 Market Impact:**")
                            impact_color = {
                                'Positive': '🟢',
                                'Negative': '🔴',
                                'Mixed': '🟡'
                            }
                            impact_emoji = impact_color.get(result.get('impact', 'N/A'), '⚪')
                            st.success(f"**Impact:** {impact_emoji} {result.get('impact', 'N/A')}")
                            st.info(f"**Reason:** {result.get('reason', 'N/A')}")
                        
                        # Detailed analysis
                        st.subheader("🔍 Detailed Analysis")
                        
                        # Sentiment analysis
                        sentiment = TextBlob(article_text).sentiment
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Sentiment Polarity", f"{sentiment.polarity:.3f}", help="Range: -1 (negative) to 1 (positive)")
                        with col2:
                            st.metric("Subjectivity", f"{sentiment.subjectivity:.3f}", help="Range: 0 (objective) to 1 (subjective)")
                        with col3:
                            word_count = len(article_text.split())
                            st.metric("Word Count", word_count)
                        
                        # Show the analyzed article
                        st.subheader("📄 Analyzed Article")
                        with st.expander("View Article Details", expanded=True):
                            st.markdown(f"**Content:** {article_text}")
                            st.markdown(f"**Analysis Summary:**")
                            st.markdown(f"- **Category:** {result['predicted_category']}")
                            st.markdown(f"- **Sector:** {result.get('sector', 'N/A')}")
                            st.markdown(f"- **Impact:** {result.get('impact', 'N/A')}")
                            st.markdown(f"- **Reasoning:** {result.get('reason', 'N/A')}")
                        
                        st.success("✅ Analysis completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
    elif input_method == "Fetch from NewsAPI":
        st.subheader("🌐 Fetch News from NewsAPI")
        
        col1, col2 = st.columns(2)
        with col1:
            query = st.text_input("Search Query:", value="business technology")
        with col2:
            max_articles = st.number_input("Max Articles:", 1, 10, 3)
        
        if st.button("📰 Fetch & Analyze News", type="primary", use_container_width=True):
            with st.spinner("Fetching news..."):
                try:
                    if fetcher:
                        articles = fetcher.fetch_news(query=query, page_size=max_articles)
                        if articles:
                            df = fetcher.articles_to_dataframe(articles)
                            
                            # Use final_mkt_analyzer for analysis
                            from final_mkt_analyzer import enrich_dataset
                            df = enrich_dataset(df)
                            
                            # Add predicted_category if missing
                            if 'predicted_category' not in df.columns:
                                df['predicted_category'] = 'business'
                            
                            # Store for visualization
                            st.session_state.enriched_data = df
                            
                            # Display results
                            st.subheader("📊 Fetched News Analysis")
                            
                            for i, (_, row) in enumerate(df.iterrows()):
                                with st.expander(f"Article {i+1}: {row['headlines'][:50]}..."):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown(f"**Headline:** {row['headlines']}")
                                        st.markdown(f"**Source:** {row.get('source', 'N/A')}")
                                        st.markdown(f"**Category:** {row['predicted_category']}")
                                    
                                    with col2:
                                        impact_emoji = {'Positive': '🟢', 'Negative': '🔴', 'Mixed': '🟡'}.get(row.get('impact', 'N/A'), '⚪')
                                        st.markdown(f"**Impact:** {impact_emoji} {row.get('impact', 'N/A')}")
                                        st.markdown(f"**Sector:** {row.get('sector', 'N/A')}")
                                        st.markdown(f"**Reason:** {row.get('reason', 'N/A')}")
                                    
                                    st.markdown(f"**Content:** {row['content'][:300]}...")
                        else:
                            st.error("❌ No articles found")
                    else:
                        st.error("❌ News fetcher not available")
                except Exception as e:
                    st.error(f"❌ Error fetching news: {e}")

# Tab 2: Model Training & Metrics
with tab2:
    st.markdown('<h2 class="section-header">🤖 Model Training & Metrics</h2>', unsafe_allow_html=True)
    
    # Load and combine datasets
    datasets = []
    dataset_info = []
    
    # Load large training dataset if available
    try:
        large_df = pd.read_csv("business_tech_trainset.csv")
        datasets.append(large_df)
        dataset_info.append(f"Large Dataset: {len(large_df)} articles")
    except:
        try:
            large_df = pd.read_csv("business_data.csv")
            from final_mkt_analyzer import enrich_dataset
            large_df = enrich_dataset(large_df)
            datasets.append(large_df)
            dataset_info.append(f"Business Data: {len(large_df)} articles")
        except:
            pass
    
    # Add user input data if available
    if st.session_state.enriched_data is not None:
        datasets.append(st.session_state.enriched_data)
        dataset_info.append(f"User Input: {len(st.session_state.enriched_data)} articles")
    
    if not datasets:
        st.error("❌ No data available. Please analyze some text first.")
        st.stop()
    
    # Combine all datasets
    df = pd.concat(datasets, ignore_index=True)
    
    # Show dataset composition
    st.info(f"📊 Combined Dataset: {len(df)} total articles")
    for info in dataset_info:
        st.write(f"  • {info}")
    
    # Model training section
    st.subheader("🏋️ Train Machine Learning Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Training Configuration:**")
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random State", 1, 100, 42)
        n_estimators = st.slider("Number of Estimators", 50, 200, 100, 10)
    
    with col2:
        st.markdown("**Model Type:**")
        model_type = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "SVM"])
        use_smote = st.checkbox("Use SMOTE for Balancing", value=True)
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            try:
                # Prepare data for training - use all categories if business/tech not enough
                business_tech = df[df['predicted_category'].isin(['business', 'technology'])]
                
                if len(business_tech) < 10:
                    # Use all data if not enough business/tech articles
                    business_tech = df
                    st.info(f"Using all {len(business_tech)} articles for training (expanded from business/tech only)")
                
                if len(business_tech) < 5:
                    st.error("Not enough articles for training (minimum 5 required)")
                else:
                    # Save updated dataset for future use
                    try:
                        df.to_csv("business_tech_trainset.csv", index=False)
                        st.success(f"💾 Dataset updated with {len(df)} articles")
                    except:
                        pass
                    # Ensure required columns exist
                    if 'impact' not in business_tech.columns:
                        business_tech['impact'] = 'Mixed'
                    
                    # Prepare features and target
                    try:
                        business_tech['cleaned_text'] = business_tech['content'].apply(lambda x: str(x).lower())
                    except:
                        from final_mkt_analyzer import clean_text
                        business_tech['cleaned_text'] = business_tech['content'].apply(clean_text)
                    
                    # TF-IDF Vectorization
                    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                    X = vectorizer.fit_transform(business_tech['cleaned_text'])
                    y = business_tech['impact']
                    
                    # Handle class imbalance with SMOTE (only if enough samples)
                    min_class_size = y.value_counts().min()
                    if use_smote and min_class_size >= 6 and len(business_tech) >= 12:
                        try:
                            smote = SMOTE(random_state=random_state)
                            X_resampled, y_resampled = smote.fit_resample(X, y)
                        except ValueError:
                            X_resampled, y_resampled = X, y
                            st.info("⚠️ SMOTE failed. Using original data.")
                    else:
                        X_resampled, y_resampled = X, y
                        if use_smote:
                            st.info(f"⚠️ Not enough data for SMOTE (min class: {min_class_size}, need 6+). Using original data.")
                    
                    # Split data (disable stratify if not enough samples)
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_resampled, y_resampled, test_size=test_size, random_state=random_state, stratify=y_resampled
                        )
                    except ValueError:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_resampled, y_resampled, test_size=test_size, random_state=random_state
                        )
                        st.info("⚠️ Using simple train-test split due to insufficient data for stratification.")
                    
                    # Train model
                    if model_type == "Random Forest":
                        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                    elif model_type == "Gradient Boosting":
                        from sklearn.ensemble import GradientBoostingClassifier
                        model = GradientBoostingClassifier(random_state=random_state)
                    else:  # SVM
                        from sklearn.svm import SVC
                        model = SVC(random_state=random_state, probability=True)
                    
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Store metrics
                    st.session_state.model_metrics = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba,
                        'model': model,
                        'vectorizer': vectorizer,
                        'dataset_size': len(df),
                        'training_size': X_train.shape[0],
                        'test_size': X_test.shape[0]
                    }
                    st.session_state.model_trained = True
                    
                    st.success("✅ Model training completed!")
                    
                    # Show dataset performance breakdown
                    st.subheader("📈 Dataset Performance Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Training Samples", X_train.shape[0])
                        st.metric("Test Samples", X_test.shape[0])
                    with col2:
                        st.metric("Total Features", X.shape[1])
                        st.metric("Impact Classes", len(set(y)))
                    
                    # Show class distribution
                    class_dist = pd.Series(y_resampled).value_counts()
                    st.write("**Class Distribution (After Processing):**")
                    for class_name, count in class_dist.items():
                        st.write(f"  • {class_name}: {count} samples")
                    
            except Exception as e:
                st.error(f"❌ Error during training: {e}")
    
    # Display model metrics
    if st.session_state.model_trained:
        st.subheader("📊 Model Performance Metrics")
        
        metrics = st.session_state.model_metrics
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1']:.4f}")
        
        # Classification Report
        st.subheader("📋 Classification Report")
        report = classification_report(metrics['y_test'], metrics['y_pred'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # Enhanced Performance Visualizations
        st.subheader("📈 Enhanced Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted comparison
            comparison_df = pd.DataFrame({
                'Actual': metrics['y_test'],
                'Predicted': metrics['y_pred']
            })
            
            actual_counts = comparison_df['Actual'].value_counts()
            predicted_counts = comparison_df['Predicted'].value_counts()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Actual',
                x=actual_counts.index,
                y=actual_counts.values,
                marker_color='#2E86AB'
            ))
            fig.add_trace(go.Bar(
                name='Predicted',
                x=predicted_counts.index,
                y=predicted_counts.values,
                marker_color='#A23B72'
            ))
            
            fig.update_layout(
                title='Actual vs Predicted Distribution',
                xaxis_title='Impact Type',
                yaxis_title='Count',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature importance (for Random Forest)
            if hasattr(metrics['model'], 'feature_importances_'):
                importances = metrics['model'].feature_importances_
                top_features = sorted(zip(importances, range(len(importances))), reverse=True)[:10]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"Feature_{i}" for _, i in top_features],
                        y=[imp for imp, _ in top_features],
                        marker_color='#F18F01'
                    )
                ])
                fig.update_layout(
                    title='Top 10 Feature Importance',
                    xaxis_title='Features',
                    yaxis_title='Importance',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Dataset composition analysis with combined data
        st.subheader("📊 Combined Dataset Analysis")
        
        # Ensure required columns exist
        if 'impact' not in df.columns:
            df['impact'] = 'Mixed'
        if 'sector' not in df.columns:
            df['sector'] = 'General'
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Category distribution
            cat_dist = df['predicted_category'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=cat_dist.index, values=cat_dist.values, hole=0.3)])
            fig.update_layout(title=f"Categories ({len(df)} articles)", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Impact distribution
            impact_dist = df['impact'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=impact_dist.index, values=impact_dist.values, hole=0.3)])
            fig.update_layout(title="Impact Distribution", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Sector distribution
            sector_dist = df['sector'].value_counts().head(5)
            fig = go.Figure(data=[go.Pie(labels=sector_dist.index, values=sector_dist.values, hole=0.3)])
            fig.update_layout(title="Top 5 Sectors", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Dataset source breakdown
        st.subheader("📈 Dataset Sources")
        source_info = []
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            for source, count in source_counts.items():
                source_info.append({'Source': source, 'Articles': count, 'Percentage': f"{count/len(df)*100:.1f}%"})
        else:
            source_info = [{'Source': 'Combined Dataset', 'Articles': len(df), 'Percentage': '100.0%'}]
        
        st.dataframe(pd.DataFrame(source_info), use_container_width=True)
    
        # Auto-retrain suggestion
        st.subheader("🔄 Continuous Learning")
        if st.button("🚀 Retrain with All Data", type="secondary"):
            st.rerun()
    
    else:
        st.info("ℹ️ Train a model to see performance metrics here.")
        
        # Show available data for training
        if len(df) > 0:
            st.subheader("📊 Available Training Data")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Articles", len(df))
            with col2:
                if 'predicted_category' in df.columns:
                    st.metric("Categories", len(df['predicted_category'].unique()))
            with col3:
                if 'impact' in df.columns:
                    st.metric("Impact Types", len(df['impact'].unique()))
    
    # Dynamic Prediction Section
    if st.session_state.model_trained and st.session_state.enriched_data is not None:
        st.subheader("🔮 Dynamic Article Prediction")
        
        # Show current analyzed articles
        current_df = st.session_state.enriched_data
        
        if len(current_df) > 0:
            st.markdown("**📊 Current Analyzed Articles:**")
            
            # Display each article with its predictions
            for i, (_, row) in enumerate(current_df.iterrows()):
                with st.expander(f"Article {i+1}: {str(row.get('headlines', 'Untitled'))[:50]}..."):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**📋 Analysis:**")
                        st.info(f"Category: {row.get('predicted_category', 'N/A')}")
                        st.info(f"Sector: {row.get('sector', 'N/A')}")
                    
                    with col2:
                        st.markdown("**📈 Impact:**")
                        impact = row.get('impact', 'N/A')
                        impact_emoji = {'Positive': '🟢', 'Negative': '🔴', 'Mixed': '🟡'}.get(impact, '⚪')
                        st.success(f"{impact_emoji} {impact}")
                    
                    with col3:
                        st.markdown("**💡 Reasoning:**")
                        st.info(f"{row.get('reason', 'N/A')}")
                    
                    # Show content preview
                    content = str(row.get('content', ''))
                    if len(content) > 200:
                        st.markdown(f"**Content Preview:** {content[:200]}...")
                    else:
                        st.markdown(f"**Content:** {content}")
            
            # Generate dynamic predictions for new text
            st.markdown("**🔮 Test New Article:**")
            new_text = st.text_area("Enter new article text for prediction:", height=150)
            
            if st.button("🎯 Predict Impact", type="secondary"):
                if new_text.strip():
                    with st.spinner("Generating prediction..."):
                        try:
                            # Create new article dataframe
                            new_article = pd.DataFrame([{
                                'headlines': 'New Article',
                                'content': new_text,
                                'url': '',
                                'source': 'User Input'
                            }])
                            
                            # Use final_mkt_analyzer for analysis
                            from final_mkt_analyzer import enrich_dataset
                            new_article = enrich_dataset(new_article)
                            
                            # Add predicted_category if missing
                            if 'predicted_category' not in new_article.columns:
                                new_article['predicted_category'] = 'business'
                            
                            # Get prediction from trained model
                            if st.session_state.model_trained:
                                metrics = st.session_state.model_metrics
                                model = metrics['model']
                                vectorizer = metrics['vectorizer']
                                
                                # Prepare text for prediction
                                from final_mkt_analyzer import clean_text
                                cleaned_text = clean_text(new_text)
                                X_new = vectorizer.transform([cleaned_text])
                                
                                # Get prediction and probability
                                prediction = model.predict(X_new)[0]
                                probabilities = model.predict_proba(X_new)[0]
                                classes = model.classes_
                                
                                # Display results
                                st.subheader("🎯 Prediction Results")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**📊 Model Prediction:**")
                                    pred_emoji = {'Positive': '🟢', 'Negative': '🔴', 'Mixed': '🟡'}.get(prediction, '⚪')
                                    st.success(f"{pred_emoji} **{prediction}**")
                                    
                                    # Show confidence scores
                                    st.markdown("**🎯 Confidence Scores:**")
                                    for i, (class_name, prob) in enumerate(zip(classes, probabilities)):
                                        st.metric(f"{class_name}", f"{prob:.3f}")
                                
                                with col2:
                                    st.markdown("**🔍 Rule-based Analysis:**")
                                    result = new_article.iloc[0]
                                    rule_emoji = {'Positive': '🟢', 'Negative': '🔴', 'Mixed': '🟡'}.get(result.get('impact', 'N/A'), '⚪')
                                    st.info(f"{rule_emoji} **{result.get('impact', 'N/A')}**")
                                    st.info(f"Sector: {result.get('sector', 'N/A')}")
                                    st.info(f"Reason: {result.get('reason', 'N/A')}")
                                
                                # Compare predictions
                                st.markdown("**⚖️ Prediction Comparison:**")
                                comparison_df = pd.DataFrame({
                                    'Method': ['ML Model', 'Rule-based'],
                                    'Prediction': [prediction, result.get('impact', 'N/A')],
                                    'Confidence': [f"{max(probabilities):.3f}", "N/A"]
                                })
                                st.dataframe(comparison_df, use_container_width=True)
                                
                                # Show why the prediction was made
                                st.markdown("**🧠 Analysis Details:**")
                                sentiment = TextBlob(new_text).sentiment
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Sentiment Polarity", f"{sentiment.polarity:.3f}")
                                with col2:
                                    st.metric("Subjectivity", f"{sentiment.subjectivity:.3f}")
                                with col3:
                                    word_count = len(new_text.split())
                                    st.metric("Word Count", word_count)
                                
                        except Exception as e:
                            st.error(f"❌ Error generating prediction: {e}")
                else:
                    st.warning("⚠️ Please enter some text for prediction.")

# Tab 3: Visualizations
with tab3:
    st.markdown('<h2 class="section-header">📈 Dynamic Visualizations</h2>', unsafe_allow_html=True)
    
    # Load and combine all datasets for visualization
    viz_datasets = []
    
    # Load large dataset
    try:
        large_df = pd.read_csv("business_tech_trainset.csv")
        viz_datasets.append(large_df)
    except:
        try:
            large_df = pd.read_csv("business_data.csv")
            from final_mkt_analyzer import enrich_dataset
            large_df = enrich_dataset(large_df)
            viz_datasets.append(large_df)
        except:
            pass
    
    # Add user input data
    if st.session_state.enriched_data is not None:
        viz_datasets.append(st.session_state.enriched_data)
    
    if viz_datasets:
        df = pd.concat(viz_datasets, ignore_index=True)
        
        # Ensure required columns exist
        if 'impact' not in df.columns:
            df['impact'] = 'Mixed'
        if 'sector' not in df.columns:
            df['sector'] = 'General'
        if 'predicted_category' not in df.columns:
            df['predicted_category'] = 'business'
        
        # Show comprehensive data summary
        st.subheader("📊 Combined Dataset Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Articles", len(df))
        with col2:
            categories = df['predicted_category'].value_counts()
            st.metric("Categories", len(categories))
        with col3:
            sectors = df['sector'].value_counts()
            st.metric("Sectors", len(sectors))
        with col4:
            impacts = df['impact'].value_counts()
            st.metric("Impact Types", len(impacts))
        
        # Show data sources
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            st.write("**Data Sources:**")
            for source, count in source_counts.items():
                st.write(f"  • {source}: {count} articles ({count/len(df)*100:.1f}%)")
        
        # Dynamic visualization options
        st.subheader("🎨 Dynamic Visualization Options")
        
        viz_options = st.multiselect(
            "Select visualizations to display:",
            ["Category Distribution", "Impact Analysis", "Sector Performance", "Sentiment Analysis", "Word Cloud", "Custom Analysis"],
            default=["Category Distribution", "Impact Analysis"]
        )
        
        # Auto-generate visualizations based on current data
        st.subheader("📈 Live Visualizations")
        
        try:
            # Category Distribution - Always show if data exists
            if "Category Distribution" in viz_options or len(viz_options) == 0:
                st.subheader("📊 Category Distribution")
                
                category_counts = df['predicted_category'].value_counts()
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=category_counts.index,
                        values=category_counts.values,
                        hole=0.3,
                        textinfo='label+percent',
                        marker=dict(colors=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
                    )
                ])
                fig.update_layout(
                    title="Article Categories Distribution",
                    showlegend=True,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Impact Analysis - Dynamic based on current data
            if "Impact Analysis" in viz_options:
                st.subheader("📈 Impact Analysis Dashboard")
                
                impact_counts = df['impact'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Impact distribution
                    fig = go.Figure(data=[
                        go.Bar(
                            x=impact_counts.index,
                            y=impact_counts.values,
                            marker_color=['#2E86AB' if x == 'Positive' else '#A23B72' if x == 'Negative' else '#F18F01' for x in impact_counts.index]
                        )
                    ])
                    fig.update_layout(
                        title="Impact Distribution",
                        xaxis_title="Impact Type",
                        yaxis_title="Count",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Sector vs Impact heatmap
                    if 'sector' in df.columns and 'impact' in df.columns:
                        pivot_data = df.groupby(['sector', 'impact']).size().unstack(fill_value=0)
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=pivot_data.values,
                            x=pivot_data.columns,
                            y=pivot_data.index,
                            colorscale='Blues',
                            text=pivot_data.values,
                            texttemplate="%{text}",
                            textfont={"size": 10}
                        ))
                        fig.update_layout(
                            title="Sector vs Impact Heatmap",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Sector Performance - Dynamic based on current data
            if "Sector Performance" in viz_options:
                st.subheader("🏭 Sector Performance")
                
                if 'sector' in df.columns:
                    sector_counts = df['sector'].value_counts()
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=sector_counts.values,
                            y=sector_counts.index,
                            orientation='h',
                            marker_color='#2E86AB'
                        )
                    ])
                    fig.update_layout(
                        title="Articles by Sector",
                        xaxis_title="Number of Articles",
                        yaxis_title="Sector",
                        height=max(400, len(sector_counts) * 30)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment Analysis - Dynamic based on current data
            if "Sentiment Analysis" in viz_options:
                st.subheader("📊 Sentiment Analysis")
                
                # Calculate sentiment for each article
                sentiments = []
                for _, row in df.iterrows():
                    sentiment = TextBlob(str(row['content'])).sentiment
                    sentiments.append({
                        'polarity': sentiment.polarity,
                        'subjectivity': sentiment.subjectivity,
                        'impact': row.get('impact', 'N/A')
                    })
                
                sentiment_df = pd.DataFrame(sentiments)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Polarity distribution
                    fig = go.Figure(data=[
                        go.Histogram(
                            x=sentiment_df['polarity'],
                            nbinsx=20,
                            marker_color='#2E86AB'
                        )
                    ])
                    fig.update_layout(
                        title="Sentiment Polarity Distribution",
                        xaxis_title="Polarity (-1 to 1)",
                        yaxis_title="Count",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Polarity vs Impact
                    fig = go.Figure()
                    for impact in sentiment_df['impact'].unique():
                        impact_data = sentiment_df[sentiment_df['impact'] == impact]
                        fig.add_trace(go.Scatter(
                            x=impact_data['polarity'],
                            y=impact_data['subjectivity'],
                            mode='markers',
                            name=impact,
                            marker=dict(size=8)
                        ))
                    fig.update_layout(
                        title="Sentiment Polarity vs Subjectivity by Impact",
                        xaxis_title="Polarity",
                        yaxis_title="Subjectivity",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Word Cloud - Dynamic based on current data
            if "Word Cloud" in viz_options:
                st.subheader("🔤 Dynamic Word Cloud")
                
                # Combine all text
                all_text = ' '.join(df['content'].astype(str))
                
                # Create word cloud
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='viridis',
                    max_words=100
                ).generate(all_text)
                
                # Convert to image
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Most Frequent Words in Analyzed Articles', fontsize=16, pad=20)
                
                st.pyplot(fig)
            
            # Custom Analysis - Dynamic insights
            if "Custom Analysis" in viz_options:
                st.subheader("🔍 Custom Dynamic Analysis")
                
                # Generate insights based on current data
                insights = []
                
                # Category insights
                top_category = df['predicted_category'].value_counts().index[0]
                insights.append(f"📊 Most analyzed category: **{top_category}** ({df['predicted_category'].value_counts().iloc[0]} articles)")
                
                # Impact insights
                if 'impact' in df.columns:
                    top_impact = df['impact'].value_counts().index[0]
                    insights.append(f"📈 Most common impact: **{top_impact}** ({df['impact'].value_counts().iloc[0]} articles)")
                
                # Sector insights
                if 'sector' in df.columns:
                    top_sector = df['sector'].value_counts().index[0]
                    insights.append(f"🏭 Most analyzed sector: **{top_sector}** ({df['sector'].value_counts().iloc[0]} articles)")
                
                # Sentiment insights
                avg_polarity = np.mean([TextBlob(str(content)).sentiment.polarity for content in df['content']])
                if avg_polarity > 0.1:
                    sentiment_trend = "Positive"
                elif avg_polarity < -0.1:
                    sentiment_trend = "Negative"
                else:
                    sentiment_trend = "Neutral"
                insights.append(f"😊 Overall sentiment trend: **{sentiment_trend}** (avg polarity: {avg_polarity:.3f})")
                
                # Display insights
                for insight in insights:
                    st.info(insight)
                
                # Show data table
                st.subheader("📋 Detailed Data View")
                display_df = df[['headlines', 'predicted_category', 'sector', 'impact', 'reason']].copy()
                st.dataframe(display_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error generating visualizations: {e}")
    
    else:
        st.info("ℹ️ No data available for visualization.")
        st.markdown("""
        **💡 How to get data:**
        1. Go to **Text Analysis** tab and analyze articles
        2. Or ensure training datasets exist (business_data.csv or business_tech_trainset.csv)
        """)

# Tab 4: Project Documentation
with tab4:
    st.markdown('<h2 class="section-header">📚 Project Documentation</h2>', unsafe_allow_html=True)
    
    # Project Overview
    st.subheader("🎯 Project Overview")
    st.markdown("""
    This News Analysis Dashboard is a comprehensive machine learning pipeline that:
    - **Categorizes** news articles into business, politics, sports, entertainment, and technology
    - **Analyzes** market impact for business and technology news
    - **Predicts** market sentiment and impact using advanced ML models
    - **Visualizes** results with interactive charts and graphs
    """)
    
    # Technical Architecture
    st.subheader("🏗️ Technical Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Data Processing Pipeline:**")
        st.markdown("""
        1. **Text Input**: Paste article or fetch from NewsAPI
        2. **Text Preprocessing**: Cleaning, tokenization, stopword removal
        3. **Feature Engineering**: TF-IDF vectorization, text normalization
        4. **Categorization**: Rule-based + ML classification
        5. **Impact Analysis**: Sector detection, sentiment analysis
        6. **Model Training**: Random Forest, SMOTE balancing
        7. **Visualization**: Interactive Plotly charts
        """)
    
    with col2:
        st.markdown("**🔧 Key Technologies:**")
        st.markdown("""
        - **Python 3.13**: Core programming language
        - **Streamlit**: Web application framework
        - **Scikit-learn**: Machine learning library
        - **Plotly**: Interactive visualizations
        - **NLTK**: Natural language processing
        - **Pandas**: Data manipulation
        - **NewsAPI**: Real-time news fetching
        - **SMOTE**: Class imbalance handling
        - **TF-IDF**: Text vectorization
        """)
    
    # Machine Learning Pipeline
    st.subheader("🤖 Machine Learning Pipeline")
    
    st.markdown("""
    **📈 Model Training Process:**
    
    1. **Text Vectorization (TF-IDF)**:
       - Converts text to numerical features
       - Max features: 1000
       - Removes English stopwords
       - Handles text normalization
    
    2. **Class Balancing (SMOTE)**:
       - Synthetic Minority Oversampling Technique
       - Addresses class imbalance in impact categories
       - Generates synthetic samples for minority classes
    
    3. **Model Selection**:
       - **Random Forest**: Ensemble method with 100 estimators
       - **Gradient Boosting**: Sequential learning approach
       - **SVM**: Support Vector Machine with probability estimates
    
    4. **Evaluation Metrics**:
       - Accuracy, Precision, Recall, F1-Score
       - Confusion Matrix visualization
       - Classification report with detailed metrics
    """)
    
    # Usage Instructions
    st.subheader("📖 Usage Instructions")
    
    st.markdown("""
    **🚀 Getting Started:**
    
    1. **Analyze Text**: Use the "Text Analysis" tab to paste articles or fetch from NewsAPI
    2. **Train Model**: Go to "Model Training & Metrics" to train ML models and view performance
    3. **View Visualizations**: Generate interactive charts in the "Visualizations" tab
    4. **Explore Documentation**: Learn about the technical implementation in this tab
    
    **💡 Tips for Best Results:**
    - Paste complete article text for accurate analysis
    - Business and technology articles work best for impact analysis
    - Train models with sufficient data for better performance
    - Use the visualizations to understand data patterns
    """)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #6C757D;">📊 News Analysis Dashboard | Built with Streamlit & Machine Learning</p>',
    unsafe_allow_html=True
)
