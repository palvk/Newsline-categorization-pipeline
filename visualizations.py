# -*- coding: utf-8 -*-
"""
Dynamic Visualizations for News Analysis System
Creates interactive charts and graphs that update based on input data
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import base64
from io import BytesIO
import json
from datetime import datetime, timedelta

class NewsVisualizer:
    def __init__(self):
        self.color_palette = {
            'business': '#2E86AB',
            'technology': '#A23B72',
            'politics': '#F18F01',
            'sports': '#C73E1D',
            'entertainment': '#7209B7',
            'positive': '#28A745',
            'negative': '#DC3545',
            'mixed': '#FFC107'
        }
        
        self.sector_colors = {
            'Technology': '#FF6B6B',
            'Manufacturing': '#4ECDC4',
            'Financial Services': '#45B7D1',
            'Energy': '#96CEB4',
            'Healthcare': '#FFEAA7',
            'Infrastructure': '#DDA0DD',
            'Retail': '#98D8C8',
            'Aviation': '#F7DC6F',
            'Macro Economy': '#BB8FCE'
        }
    
    def create_category_distribution(self, df):
        """Create interactive pie chart for news categories"""
        if 'predicted_category' not in df.columns:
            return None
        
        category_counts = df['predicted_category'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=category_counts.index,
            values=category_counts.values,
            marker_colors=[self.color_palette.get(cat, '#6C757D') for cat in category_counts.index],
            textinfo='label+percent+value',
            textfont_size=12,
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': '📊 News Category Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2C3E50'}
            },
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.01
            ),
            margin=dict(l=20, r=20, t=60, b=20),
            height=500
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="category-chart")
    
    def create_impact_analysis(self, df):
        """Create impact analysis visualization"""
        business_tech = df[df['predicted_category'].isin(['business', 'technology'])]
        
        if len(business_tech) == 0 or 'impact' not in business_tech.columns:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Impact Distribution', 'Sector Analysis', 'Impact by Sector', 'Timeline'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Impact Distribution Pie Chart
        impact_counts = business_tech['impact'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=impact_counts.index,
                values=impact_counts.values,
                marker_colors=[self.color_palette.get(imp, '#6C757D') for imp in impact_counts.index],
                name="Impact",
                textinfo='label+percent'
            ),
            row=1, col=1
        )
        
        # 2. Sector Analysis Bar Chart
        if 'sector' in business_tech.columns:
            sector_counts = business_tech['sector'].value_counts().head(8)
            fig.add_trace(
                go.Bar(
                    x=sector_counts.values,
                    y=sector_counts.index,
                    orientation='h',
                    marker_color=[self.sector_colors.get(sec, '#6C757D') for sec in sector_counts.index],
                    name="Sectors",
                    text=sector_counts.values,
                    textposition='auto'
                ),
                row=1, col=2
            )
        
        # 3. Impact by Sector
        if 'sector' in business_tech.columns and 'impact' in business_tech.columns:
            impact_sector = pd.crosstab(business_tech['sector'], business_tech['impact'])
            for impact in impact_sector.columns:
                fig.add_trace(
                    go.Bar(
                        x=impact_sector.index,
                        y=impact_sector[impact],
                        name=impact,
                        marker_color=self.color_palette.get(impact, '#6C757D')
                    ),
                    row=2, col=1
                )
        
        # 4. Timeline (if published_at available)
        if 'published_at' in business_tech.columns:
            business_tech['date'] = pd.to_datetime(business_tech['published_at']).dt.date
            timeline = business_tech.groupby(['date', 'impact']).size().reset_index(name='count')
            
            for impact in timeline['impact'].unique():
                impact_data = timeline[timeline['impact'] == impact]
                fig.add_trace(
                    go.Scatter(
                        x=impact_data['date'],
                        y=impact_data['count'],
                        mode='lines+markers',
                        name=f'{impact} Timeline',
                        line=dict(color=self.color_palette.get(impact, '#6C757D'))
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title={
                'text': '📈 Market Impact Analysis Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2C3E50'}
            },
            height=800,
            showlegend=True,
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="impact-dashboard")
    
    def create_word_cloud(self, df, category=None):
        """Create word cloud from news content"""
        if category:
            text_data = df[df['predicted_category'] == category]['content'].str.cat(sep=' ')
        else:
            text_data = df['content'].str.cat(sep=' ')
        
        if not text_data or len(text_data.strip()) == 0:
            return None
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate(text_data)
        
        # Convert to base64 for HTML embedding
        img_buffer = BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f'<img src="data:image/png;base64,{img_base64}" style="width:100%; height:auto;">'
    
    def create_sentiment_trend(self, df):
        """Create sentiment trend analysis"""
        business_tech = df[df['predicted_category'].isin(['business', 'technology'])]
        
        if len(business_tech) == 0:
            return None
        
        # Create sentiment score based on impact
        sentiment_map = {'Positive': 1, 'Negative': -1, 'Mixed': 0}
        business_tech['sentiment_score'] = business_tech['impact'].map(sentiment_map)
        
        # Group by date if available
        if 'published_at' in business_tech.columns:
            business_tech['date'] = pd.to_datetime(business_tech['published_at']).dt.date
            daily_sentiment = business_tech.groupby('date')['sentiment_score'].mean().reset_index()
            
            fig = go.Figure()
            
            # Add sentiment line
            fig.add_trace(go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['sentiment_score'],
                mode='lines+markers',
                name='Sentiment Trend',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8)
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig.update_layout(
                title={
                    'text': '📊 Market Sentiment Trend',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#2C3E50'}
                },
                xaxis_title='Date',
                yaxis_title='Sentiment Score',
                yaxis=dict(
                    tickmode='array',
                    tickvals=[-1, 0, 1],
                    ticktext=['Negative', 'Mixed', 'Positive']
                ),
                height=400,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            return fig.to_html(include_plotlyjs='cdn', div_id="sentiment-trend")
        
        return None
    
    def create_sector_performance(self, df):
        """Create sector performance comparison"""
        business_tech = df[df['predicted_category'].isin(['business', 'technology'])]
        
        if len(business_tech) == 0 or 'sector' not in business_tech.columns or 'impact' not in business_tech.columns:
            return None
        
        # Calculate sector performance
        sector_performance = business_tech.groupby('sector')['impact'].apply(
            lambda x: (x == 'Positive').sum() / len(x) * 100
        ).reset_index(name='positive_percentage')
        
        sector_performance = sector_performance.sort_values('positive_percentage', ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=sector_performance['positive_percentage'],
                y=sector_performance['sector'],
                orientation='h',
                marker_color=[self.sector_colors.get(sec, '#6C757D') for sec in sector_performance['sector']],
                text=[f"{val:.1f}%" for val in sector_performance['positive_percentage']],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Positive Impact: %{x:.1f}%<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title={
                'text': '🏭 Sector Performance Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2C3E50'}
            },
            xaxis_title='Positive Impact Percentage (%)',
            yaxis_title='Sector',
            height=500,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="sector-performance")
    
    def create_summary_stats(self, df):
        """Create summary statistics cards"""
        total_articles = len(df)
        business_tech = df[df['predicted_category'].isin(['business', 'technology'])]
        analyzed_articles = len(business_tech)
        
        stats = {
            'total_articles': int(total_articles),
            'analyzed_articles': int(analyzed_articles),
            'categories': int(df['predicted_category'].nunique()) if 'predicted_category' in df.columns else 0,
            'sectors': int(business_tech['sector'].nunique()) if 'sector' in business_tech.columns else 0
        }
        
        if 'impact' in business_tech.columns:
            impact_counts = business_tech['impact'].value_counts()
            stats.update({
                'positive_impact': int(impact_counts.get('Positive', 0)),
                'negative_impact': int(impact_counts.get('Negative', 0)),
                'mixed_impact': int(impact_counts.get('Mixed', 0))
            })
        
        return stats
    
    def create_dashboard(self, df):
        """Create complete dashboard with all visualizations"""
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>News Analysis Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 10px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .stat-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .stat-number {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #2E86AB;
                }}
                .stat-label {{
                    color: #6C757D;
                    margin-top: 5px;
                }}
                .chart-container {{
                    background: white;
                    margin-bottom: 30px;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .wordcloud-container {{
                    background: white;
                    margin-bottom: 30px;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .timestamp {{
                    text-align: center;
                    color: #6C757D;
                    margin-top: 20px;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 News Analysis Dashboard</h1>
                    <p>Real-time Market Impact Analysis</p>
                </div>
        """
        
        # Add summary statistics
        stats = self.create_summary_stats(df)
        dashboard_html += f"""
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{stats['total_articles']}</div>
                        <div class="stat-label">Total Articles</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['analyzed_articles']}</div>
                        <div class="stat-label">Analyzed Articles</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['categories']}</div>
                        <div class="stat-label">Categories</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['sectors']}</div>
                        <div class="stat-label">Sectors</div>
                    </div>
                </div>
        """
        
        # Add category distribution
        category_chart = self.create_category_distribution(df)
        if category_chart:
            dashboard_html += f"""
                <div class="chart-container">
                    {category_chart}
                </div>
            """
        
        # Add impact analysis
        impact_chart = self.create_impact_analysis(df)
        if impact_chart:
            dashboard_html += f"""
                <div class="chart-container">
                    {impact_chart}
                </div>
            """
        
        # Add sector performance
        sector_chart = self.create_sector_performance(df)
        if sector_chart:
            dashboard_html += f"""
                <div class="chart-container">
                    {sector_chart}
                </div>
            """
        
        # Add sentiment trend
        sentiment_chart = self.create_sentiment_trend(df)
        if sentiment_chart:
            dashboard_html += f"""
                <div class="chart-container">
                    {sentiment_chart}
                </div>
            """
        
        # Add word cloud
        wordcloud = self.create_word_cloud(df)
        if wordcloud:
            dashboard_html += f"""
                <div class="wordcloud-container">
                    <h3>🔤 Most Frequent Words</h3>
                    {wordcloud}
                </div>
            """
        
        # Add timestamp
        dashboard_html += f"""
                <div class="timestamp">
                    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """
        
        return dashboard_html

def main():
    """Test the visualizer with sample data"""
    # Load test data
    try:
        df = pd.read_csv('test_results.csv')
        visualizer = NewsVisualizer()
        
        # Create dashboard
        dashboard = visualizer.create_dashboard(df)
        
        # Save dashboard
        with open('dashboard.html', 'w', encoding='utf-8') as f:
            f.write(dashboard)
        
        print("✅ Dashboard created successfully!")
        print("📁 Open 'dashboard.html' in your browser to view the visualizations")
        
    except FileNotFoundError:
        print("❌ No test data found. Please run test_system.py first.")
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")

if __name__ == "__main__":
    main()
