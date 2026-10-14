import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.preprocessing import preprocess_text, extract_noun_phrases
from src.trend_extraction import extract_trends_from_phrases
from src.trend_analysis import calculate_trend_shifts
from src.sentiment_model import SentimentModel
from src.fine_tuned_model import FineTunedSentimentModel, is_fine_tuned_model_available


st.set_page_config(page_title="Fashion Trend shifts Dashboard", layout="wide")

# Cache data loading
@st.cache_data
def load_data():
    processed_path = "data/processed/processed_fashion_data.csv"
    if not os.path.exists(processed_path):
        return None
    df = pd.read_csv(processed_path)
    df['date'] = pd.to_datetime(df['date'])
    # Convert string back to list
    df['detected_trends'] = df['detected_trends'].apply(lambda x: x.split(',') if pd.notnull(x) and x != "" else [])
    return df

@st.cache_resource
def load_model():
    model_path = "models/sentiment_model/sentiment_pipeline.pkl"
    if not os.path.exists(model_path):
        return None
    model = SentimentModel()
    model.load(model_path)
    return model

@st.cache_resource
def load_fine_tuned_model():
    if not is_fine_tuned_model_available():
        return None
    return FineTunedSentimentModel()

def main():
    st.title("👗 Fashion Trend Shifts & Sentiment Dashboard")
    st.markdown("Analyze fashion influencer captions and audience comments using NLP (spaCy & ML).")
    
    df = load_data()
    model = load_model()
    
    if df is None:
        st.warning("Processed data not found. Please run the data preparation scripts first.")
        return
        
    # Tabs
    tabs = st.tabs([
        "Overview", "Trend Analysis", "Trend Shifts", "Sentiment Analysis", 
        "Influencer Analysis", "Text Analyzer", "Dataset Explorer", "Model Performance"
    ])
    
    # 1. Overview
    with tabs[0]:
        st.header("Project Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Posts/Comments", len(df))
        col2.metric("Total Influencers", df['influencer'].nunique())
        
        # Count total unique trends
        all_trends = [t for sublist in df['detected_trends'] for t in sublist]
        unique_trends = len(set(all_trends))
        col3.metric("Detected Trends", unique_trends)
        
        # Most popular trend
        from collections import Counter
        most_popular = Counter(all_trends).most_common(1)[0][0] if all_trends else "N/A"
        col4.metric("Top Trend", most_popular)
        
        st.subheader("Dataset Preview")
        st.dataframe(df[['date', 'influencer', 'text', 'detected_trends', 'predicted_sentiment']].head())
        
    # 2. Trend Analysis
    with tabs[1]:
        st.header("Fashion Trend Analysis & Unsupervised Topic Discovery")
        
        analysis_mode = st.radio(
            "Trend Extraction Perspective:",
            ["Rule/Seed-based Trends", "Unsupervised Discovered Topics (NMF / Topic Modeling)"],
            horizontal=True
        )
        
        if analysis_mode == "Rule/Seed-based Trends":
            if all_trends:
                trend_counts = Counter(all_trends).most_common(20)
                trend_df = pd.DataFrame(trend_counts, columns=['Trend', 'Mentions'])
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(trend_df, x='Mentions', y='Trend', orientation='h', title="Top Fashion Trends (Seed Keywords + Phrases)")
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col2:
                    st.subheader("Trend Word Cloud")
                    wordcloud = WordCloud(width=800, height=600, background_color='white').generate(" ".join(all_trends))
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
            else:
                st.info("No trends detected.")
        else:
            st.subheader("💡 Unsupervised Topic Discovery (Non-Negative Matrix Factorization)")
            st.markdown(
                "Unlike dictionary matching, **NMF decomposes the TF-IDF document-term matrix** into latent thematic topics. "
                "This allows discovering emerging slang, micro-trends, and aesthetics automatically."
            )
            
            topics_file = "reports/discovered_topics.json"
            if os.path.exists(topics_file):
                with open(topics_file, 'r') as f:
                    topics_info = json.load(f)
                
                col_t1, col_t2 = st.columns([1, 1])
                with col_t1:
                    st.markdown("#### Discovered Thematic Topics")
                    for t in topics_info.get('topics', []):
                        keywords_pill = " • ".join(t['top_words'])
                        st.markdown(f"**Topic {t['id'] + 1}**: `{keywords_pill}`")
                
                with col_t2:
                    dist = topics_info.get('topic_distribution', {})
                    if dist:
                        dist_df = pd.DataFrame(list(dist.items()), columns=['Topic', 'Post Count'])
                        fig_topics = px.bar(
                            dist_df, x='Post Count', y='Topic', orientation='h',
                            title="Post Distribution by Discovered Topic",
                            color='Post Count', color_continuous_scale='Blues'
                        )
                        fig_topics.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_topics, use_container_width=True)
            else:
                st.info("Topic discovery report not found. Run scripts/process_data.py to extract topics.")

            
    # 3. Trend Shifts
    with tabs[2]:
        st.header("Trend Shifts Over Time")
        shift_df = calculate_trend_shifts(df)
        
        if not shift_df.empty:
            trends_list = sorted(list(shift_df['detected_trends'].unique()))
            selected_trend = st.selectbox("Select a trend to analyze:", trends_list)
            
            trend_data = shift_df[shift_df['detected_trends'] == selected_trend].copy()
            trend_data['month_str'] = trend_data['month_year'].astype(str)
            
            fig = px.line(trend_data, x='month_str', y='mentions', markers=True, title=f"Mentions over time: {selected_trend}")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader(f"Current Status: {trend_data['status'].iloc[-1]}")
            st.dataframe(trend_data[['month_str', 'mentions', 'prev_mentions', 'growth_rate', 'status']])
        else:
            st.info("Not enough data for trend shifts.")
            
    # 4. Sentiment Analysis
    with tabs[3]:
        st.header("Audience Sentiment Analysis")
        sentiment_counts = df['predicted_sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(sentiment_counts, values='Count', names='Sentiment', title="Overall Sentiment Distribution", hole=0.4,
                         color='Sentiment', color_discrete_map={'positive':'green', 'neutral':'gray', 'negative':'red'})
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Sentiment by Trend")
            # Explode trends to get sentiment per trend
            exploded = df.explode('detected_trends').dropna(subset=['detected_trends'])
            sent_trend = exploded.groupby(['detected_trends', 'predicted_sentiment']).size().reset_index(name='count')
            # Get top 10 trends
            top_10 = [t[0] for t in Counter(all_trends).most_common(10)]
            sent_trend_top = sent_trend[sent_trend['detected_trends'].isin(top_10)]
            
            fig = px.bar(sent_trend_top, x='detected_trends', y='count', color='predicted_sentiment', 
                         title="Sentiment for Top 10 Trends", barmode='stack',
                         color_discrete_map={'positive':'green', 'neutral':'gray', 'negative':'red'})
            st.plotly_chart(fig, use_container_width=True)
            
    # 5. Influencer Analysis
    with tabs[4]:
        st.header("Influencer Insights")
        influencers = sorted(df['influencer'].unique())
        selected_inf = st.selectbox("Select Influencer:", influencers)
        
        inf_data = df[df['influencer'] == selected_inf]
        
        col1, col2 = st.columns(2)
        col1.metric("Total Posts/Comments", len(inf_data))
        
        inf_trends = [t for sublist in inf_data['detected_trends'] for t in sublist]
        top_inf_trend = Counter(inf_trends).most_common(1)[0][0] if inf_trends else "N/A"
        col2.metric("Favorite Trend", top_inf_trend)
        
        if inf_trends:
            inf_trend_df = pd.DataFrame(Counter(inf_trends).most_common(10), columns=['Trend', 'Mentions'])
            fig = px.bar(inf_trend_df, x='Mentions', y='Trend', orientation='h', title=f"Top Trends for {selected_inf}")
            st.plotly_chart(fig, use_container_width=True)
            
    # 6. Text Analyzer
    with tabs[5]:
        st.header("Real-time Text Analyzer")
        st.markdown("Analyze any text using either the **Pre-trained HuggingFace Model** or our **Fine-Tuned Custom Model**.")

        user_text = st.text_area(
            "Enter a fashion caption or comment:",
            "I am obsessed with these oversized denim jackets!"
        )

        fine_tuned_ready = is_fine_tuned_model_available()

        model_choice = st.radio(
            "Select Model to Use:",
            options=["Pre-trained HuggingFace (DistilBERT)", "Our Fine-Tuned Model"],
            help="The Fine-Tuned Model is only available after running scripts/fine_tune_transformer.py"
        )

        if model_choice == "Our Fine-Tuned Model" and not fine_tuned_ready:
            st.warning(
                "⚠️ Fine-tuned model not found. Please run:\n\n"
                "```\npython scripts/fine_tune_transformer.py\n```\n\n"
                "This will train and save your custom model locally."
            )
        elif st.button("Analyze Text"):
            with st.spinner("Analyzing..."):
                # --- NLP Processing (spaCy) — always runs ---
                cleaned      = preprocess_text(user_text)
                noun_phrases = extract_noun_phrases(user_text)
                trends       = extract_trends_from_phrases([noun_phrases])[0]

                # --- Sentiment Prediction ---
                if model_choice == "Our Fine-Tuned Model" and fine_tuned_ready:
                    ft_model = load_fine_tuned_model()
                    if ft_model is None:
                        st.error("Failed to load fine-tuned model.")
                        st.stop()
                    result   = ft_model.predict_single(user_text)
                    sentiment    = result["label"]
                    confidence   = result["confidence"]
                    scores       = result["scores"]
                    model_used   = "Our Fine-Tuned DistilBERT (trained on fashion dataset)"
                else:

                    # Pre-trained HuggingFace
                    if model is None:
                        st.error("Pre-trained model not loaded.")
                        st.stop()
                    sentiment  = model.predict([user_text])[0]
                    confidence = None
                    scores     = None
                    model_used = "Pre-trained HuggingFace DistilBERT"

            st.success("Analysis Complete!")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("NLP Processing (spaCy)")
                st.write(f"**Cleaned Text:** {cleaned}")
                st.write(f"**Noun Phrases:** {noun_phrases if noun_phrases else 'None'}")
                st.write(f"**Detected Fashion Trends:** {trends if trends else 'None detected'}")

            with col2:
                st.subheader("Sentiment Prediction")
                st.write(f"**Model Used:** {model_used}")

                # Color-coded sentiment badge
                color_map = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
                badge = color_map.get(sentiment, "⚪")
                st.markdown(f"## {badge} {sentiment.capitalize()}")

                if confidence is not None:
                    st.write(f"**Confidence:** {confidence}%")

                if scores is not None:
                    st.subheader("Score Breakdown")
                    score_df = pd.DataFrame([scores]).T.reset_index()
                    score_df.columns = ["Sentiment", "Score (%)"]
                    score_df = score_df.sort_values("Score (%)", ascending=False)
                    fig = px.bar(score_df, x="Sentiment", y="Score (%)",
                                 color="Sentiment",
                                 color_discrete_map={"positive": "green", "neutral": "gray", "negative": "red"},
                                 title="Prediction Confidence Breakdown")
                    st.plotly_chart(fig, use_container_width=True)



    # 7. Dataset Explorer
    with tabs[6]:
        st.header("Dataset Explorer")
        st.dataframe(df)
        
        st.subheader("Upload New Data")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file is not None:
            st.write("File uploaded successfully. Note: To process this file and update the dashboard, you would run it through the preprocessing pipeline scripts.")
            
    # 8. Model Performance
    with tabs[7]:
        st.header("Model Evaluation")
        report_path = "reports/evaluation_report.json"
        cm_path = "reports/confusion_matrix.png"
        
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
                
            st.metric("Accuracy", f"{report['accuracy']:.4f}")
            
            st.subheader("Classification Report")
            report_df = pd.DataFrame(report['classification_report']).transpose()
            st.dataframe(report_df)
            
            if os.path.exists(cm_path):
                st.subheader("Confusion Matrix")
                st.image(cm_path)
        else:
            st.info("Evaluation report not found. Run scripts/evaluate_model.py.")

if __name__ == "__main__":
    main()
