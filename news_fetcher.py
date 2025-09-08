# -*- coding: utf-8 -*-
"""
News Fetcher using NewsAPI
Fetches news articles and saves them to CSV format
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import os

class NewsFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        self.headers = {
            'X-API-Key': api_key
        }
    
    def fetch_news(self, query=None, sources=None, domains=None, 
                   from_date=None, to_date=None, language='en', 
                   sort_by='publishedAt', page_size=100):
        """
        Fetch news articles from NewsAPI
        
        Args:
            query: Keywords or phrases to search for
            sources: Comma-separated string of identifiers for news sources
            domains: Comma-separated string of domains to restrict the search
            from_date: Start date (YYYY-MM-DD format)
            to_date: End date (YYYY-MM-DD format)
            language: Language code (default: 'en')
            sort_by: Sort order ('relevancy', 'popularity', 'publishedAt')
            page_size: Number of results per page (max 100)
        """
        
        params = {
            'apiKey': self.api_key,
            'language': language,
            'sortBy': sort_by,
            'pageSize': page_size
        }
        
        if query:
            params['q'] = query
        if sources:
            params['sources'] = sources
        if domains:
            params['domains'] = domains
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        try:
            response = requests.get(self.base_url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'ok':
                return data['articles']
            else:
                print(f"API Error: {data.get('message', 'Unknown error')}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return []
    
    def fetch_multiple_queries(self, queries, max_articles_per_query=50):
        """
        Fetch news for multiple queries and combine results
        """
        all_articles = []
        
        for query in queries:
            print(f"Fetching news for: {query}")
            articles = self.fetch_news(query=query, page_size=max_articles_per_query)
            
            for article in articles:
                article['search_query'] = query
            
            all_articles.extend(articles)
            time.sleep(1)  # Rate limiting
        
        return all_articles
    
    def articles_to_dataframe(self, articles):
        """
        Convert articles list to pandas DataFrame
        """
        if not articles:
            return pd.DataFrame()
        
        # Extract relevant fields
        data = []
        for article in articles:
            data.append({
                'headlines': article.get('title', ''),
                'description': article.get('description', ''),
                'content': article.get('content', '') or article.get('description', ''),
                'url': article.get('url', ''),
                'source': article.get('source', {}).get('name', ''),
                'published_at': article.get('publishedAt', ''),
                'search_query': article.get('search_query', ''),
                'category': 'unclassified'  # Will be classified later
            })
        
        return pd.DataFrame(data)
    
    def save_to_csv(self, df, filename):
        """
        Save DataFrame to CSV file
        """
        if not df.empty:
            df.to_csv(filename, index=False)
            print(f"Saved {len(df)} articles to {filename}")
        else:
            print("No articles to save")

def main():
    # Initialize NewsAPI with your API key
    API_KEY = "17bb213791da4effb5e2ac8f0d3ef504"
    fetcher = NewsFetcher(API_KEY)
    
    # Define search queries for different categories
    queries = {
        'business': [
            'business news', 'economy', 'finance', 'stock market', 
            'corporate earnings', 'merger acquisition', 'IPO', 'startup funding'
        ],
        'technology': [
            'technology news', 'artificial intelligence', 'machine learning',
            'software development', 'cybersecurity', 'blockchain', 'cloud computing'
        ],
        'politics': [
            'politics', 'government', 'election', 'policy', 'legislation',
            'parliament', 'congress', 'senate'
        ],
        'sports': [
            'sports', 'football', 'basketball', 'cricket', 'tennis',
            'olympics', 'world cup', 'championship'
        ],
        'entertainment': [
            'entertainment', 'movies', 'music', 'celebrity', 'hollywood',
            'bollywood', 'television', 'streaming'
        ]
    }
    
    # Fetch news for each category
    all_articles = []
    
    for category, category_queries in queries.items():
        print(f"\n=== Fetching {category.upper()} news ===")
        articles = fetcher.fetch_multiple_queries(category_queries, max_articles_per_query=20)
        
        # Add category to articles
        for article in articles:
            article['category'] = category
        
        all_articles.extend(articles)
        print(f"Fetched {len(articles)} {category} articles")
    
    # Convert to DataFrame
    df = fetcher.articles_to_dataframe(all_articles)
    
    if not df.empty:
        # Save all articles
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"news_articles_{timestamp}.csv"
        fetcher.save_to_csv(df, filename)
        
        # Show category distribution
        print(f"\nCategory Distribution:")
        print(df['category'].value_counts())
        
        # Show sample data
        print(f"\nSample data:")
        print(df[['headlines', 'category', 'source']].head())
        
        return filename
    else:
        print("No articles fetched!")
        return None

if __name__ == "__main__":
    main()
