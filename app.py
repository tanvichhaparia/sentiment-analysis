import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.preprocessing import preprocess_text, extract_noun_phrases
from src.trend_extraction import extract_trends_from_phrases
from src.trend_analysis import calculate_trend_shifts
from src.sentiment_model import SentimentModel
from src.fine_tuned_model import FineTunedSentimentModel, is_fine_tuned_model_available

# Editorial Fashion Palette
# Primary background: Deep warm charcoal / noir
# Accent: Haute champagne / gold / terracotta / muted sage
FASHION_BG = "#0E0E10"
FASHION_SURFACE = "#17181C"
FASHION_BORDER = "#2A2B31"
FASHION_GOLD = "#D4AF37"
FASHION_CREAM = "#F5F5F0"
FASHION_MUTED = "#8E909A"
FASHION_ROSE = "#C57B7B"
FASHION_SAGE = "#7DA88B"

st.set_page_config(
    page_title="HAUTE METRICS | Fashion Trend & Sentiment Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# High-Fashion Editorial CSS Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Montserrat:wght@200;300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
        color: #E2E2E6;
    }

    /* Editorial Headline Styling - Sharp High-Contrast Masthead */
    .fashion-title {
        font-family: 'Cinzel', serif !important;
        font-weight: 700 !important;
        font-size: 2.6rem !important;
        letter-spacing: 0.22em !important;
        text-transform: uppercase !important;
        color: #111216 !important;
        margin-bottom: 0.2rem !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.12);
    }

    @media (prefers-color-scheme: dark) {
        .fashion-title {
            color: #F8F8F6 !important;
            text-shadow: 0 2px 10px rgba(212, 175, 55, 0.25);
        }
    }


    .fashion-subtitle {
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
        font-size: 0.88rem;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: #D4AF37;
        margin-bottom: 1.8rem;
        border-bottom: 1px solid #2A2B31;
        padding-bottom: 0.8rem;
    }

    /* KPI Editorial Card */
    .kpi-container {
        background: linear-gradient(135deg, #16171B 0%, #1A1C22 100%);
        border: 1px solid #282A32;
        border-radius: 2px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }

    .kpi-label {
        font-family: 'Montserrat', sans-serif;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.22em;
        color: #8E909A;
        margin-bottom: 0.3rem;
    }

    .kpi-value {
        font-family: 'Cinzel', serif;
        font-size: 1.85rem;
        font-weight: 600;
        color: #FAF9F6;
        letter-spacing: 0.05em;
    }

    /* Minimalist status badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border: 1px solid #D4AF37;
        color: #D4AF37;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        background: rgba(212, 175, 55, 0.08);
        border-radius: 2px;
        font-weight: 500;
    }

    .sentiment-badge-pos {
        display: inline-block;
        padding: 0.35rem 1rem;
        border: 1px solid #7DA88B;
        color: #7DA88B;
        font-size: 0.85rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        background: rgba(125, 168, 139, 0.1);
        border-radius: 2px;
    }

    .sentiment-badge-neg {
        display: inline-block;
        padding: 0.35rem 1rem;
        border: 1px solid #C57B7B;
        color: #C57B7B;
        font-size: 0.85rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        background: rgba(197, 123, 123, 0.1);
        border-radius: 2px;
    }

    .sentiment-badge-neu {
        display: inline-block;
        padding: 0.35rem 1rem;
        border: 1px solid #8E909A;
        color: #B4B6C0;
        font-size: 0.85rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        background: rgba(142, 144, 154, 0.1);
        border-radius: 2px;
    }

    .topic-pill {
        display: inline-block;
        margin: 0.25rem 0.25rem;
        padding: 0.35rem 0.8rem;
        background: #1B1D23;
        border: 1px solid #2C2F38;
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        color: #D8D9E0;
        border-radius: 2px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid #282A32;
        padding-bottom: 0.3rem;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Cinzel', serif;
        font-size: 0.82rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #8E909A;
        padding: 0.6rem 0.5rem;
        border: none;
        background: transparent;
    }

    .stTabs [aria-selected="true"] {
        color: #D4AF37 !important;
        border-bottom: 2px solid #D4AF37 !important;
    }

    /* Clean Streamlit Components */
    .stButton>button {
        background-color: #1A1C22;
        color: #FAF9F6;
        border: 1px solid #D4AF37;
        border-radius: 2px;
        font-family: 'Montserrat', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-size: 0.78rem;
        padding: 0.5rem 1.6rem;
        transition: all 0.25s ease;
    }

    .stButton>button:hover {
        background-color: #D4AF37;
        color: #0E0E10;
        border-color: #D4AF37;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Plotly High-Fashion Theme
def apply_fashion_layout(fig, title=""):
    fig.update_layout(
        title=dict(
            text=f"<span style='font-family: Cinzel; letter-spacing: 0.15em; font-size: 14px; text-transform: uppercase; color: #FAF9F6;'>{title}</span>",
            x=0.02, y=0.96
        ),
        paper_bgcolor="#16171B",
        plot_bgcolor="#16171B",
        font=dict(family="Montserrat", size=11, color="#A8AAB4"),
        margin=dict(l=30, r=30, t=50, b=30),
        xaxis=dict(gridcolor="#22242B", linecolor="#2C2F38", zeroline=False),
        yaxis=dict(gridcolor="#22242B", linecolor="#2C2F38", zeroline=False),
    )
    return fig

# Data Loader
@st.cache_data
def load_data():
    processed_path = "data/processed/processed_fashion_data.csv"
    if not os.path.exists(processed_path):
        return None
    df = pd.read_csv(processed_path)
    df['date'] = pd.to_datetime(df['date'])
    df['detected_trends'] = df['detected_trends'].apply(
        lambda x: [item.strip() for item in x.split(',') if item.strip()] if pd.notnull(x) and x != "" else []
    )
    return df

# Model Loaders
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
    # Editorial Header
    st.markdown('<div class="fashion-title">HAUTE METRICS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fashion-subtitle">Automated Text Mining & Sentiment Intelligence for Fast-Fashion Trend Shifts</div>',
        unsafe_allow_html=True
    )
    
    df = load_data()
    model = load_model()
    
    if df is None:
        st.warning("Processed dataset not located. Run scripts/process_data.py to initialize data.")
        return

    # Master Tab Navigation
    tabs = st.tabs([
        "OVERVIEW",
        "TREND ANALYSIS",
        "TEMPORAL SHIFTS",
        "SENTIMENT INTELLIGENCE",
        "INFLUENCER RADAR",
        "TEXT ANALYZER",
        "DATASET REPOSITORY",
        "MODEL PERFORMANCE"
    ])

    # 1. OVERVIEW
    with tabs[0]:
        st.markdown("### EXECUTIVE SUMMARY")
        
        all_trends = [t for sublist in df['detected_trends'] for t in sublist]
        unique_trends = len(set(all_trends))
        most_popular = Counter(all_trends).most_common(1)[0][0] if all_trends else "N/A"
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="kpi-container"><div class="kpi-label">Corpus Volume</div>'
                f'<div class="kpi-value">{len(df):,}</div></div>',
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f'<div class="kpi-container"><div class="kpi-label">Monitored Creators</div>'
                f'<div class="kpi-value">{df["influencer"].nunique()}</div></div>',
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f'<div class="kpi-container"><div class="kpi-label">Tracked Trends</div>'
                f'<div class="kpi-value">{unique_trends}</div></div>',
                unsafe_allow_html=True
            )
        with c4:
            st.markdown(
                f'<div class="kpi-container"><div class="kpi-label">Predominant Aesthetic</div>'
                f'<div class="kpi-value" style="font-size: 1.35rem;">{most_popular.upper()}</div></div>',
                unsafe_allow_html=True
            )
        
        st.markdown("#### RECENT SOCIAL RECORDS")
        display_cols = ['date', 'influencer', 'text', 'detected_trends', 'predicted_sentiment']
        clean_preview = df[display_cols].head(8).copy()
        clean_preview['detected_trends'] = clean_preview['detected_trends'].apply(lambda x: ", ".join(x) if x else "-")
        st.dataframe(clean_preview, use_container_width=True)

    # 2. TREND ANALYSIS
    with tabs[1]:
        st.markdown("### TREND TAXONOMY & DISCOVERY")
        
        analysis_mode = st.radio(
            "Analytical Perspective",
            ["Seed Taxonomy & Lexical Matching", "Unsupervised Thematic Discovery (NMF / Topic Modeling)"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if analysis_mode == "Seed Taxonomy & Lexical Matching":
            if all_trends:
                trend_counts = Counter(all_trends).most_common(15)
                trend_df = pd.DataFrame(trend_counts, columns=['Trend', 'Mentions'])
                
                col1, col2 = st.columns([1.1, 0.9])
                with col1:
                    fig = px.bar(
                        trend_df, x='Mentions', y='Trend', orientation='h',
                        color='Mentions',
                        color_continuous_scale=[[0, '#2C2F38'], [0.5, '#7A8094'], [1, '#D4AF37']]
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
                    apply_fashion_layout(fig, "Leading Fashion Trends by Volume")
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col2:
                    st.markdown("#### AESTHETIC LEXICON CLOUD")
                    wordcloud = WordCloud(
                        width=800, height=520,
                        background_color='#16171B',
                        colormap='cividis',
                        collocations=False
                    ).generate(" ".join(all_trends))
                    fig_wc, ax = plt.subplots(figsize=(8, 5.2), facecolor='#16171B')
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig_wc)
            else:
                st.info("No trends identified in the current index.")
        else:
            st.markdown("#### LATENT DIRICHLET & NON-NEGATIVE MATRIX DECOMPOSITION")
            st.caption("Unsupervised factorization extracts emerging micro-trends and slang directly from TF-IDF n-grams without manual seed dictionaries.")
            
            topics_file = "reports/discovered_topics.json"
            if os.path.exists(topics_file):
                with open(topics_file, 'r') as f:
                    topics_info = json.load(f)
                
                col_t1, col_t2 = st.columns([1, 1.1])
                with col_t1:
                    st.markdown("#### DISCOVERED LATENT TOPICS")
                    for t in topics_info.get('topics', []):
                        keywords_pill = "  |  ".join(t['top_words'][:5])
                        st.markdown(
                            f'<div class="kpi-container" style="padding: 0.8rem 1rem; margin-bottom: 0.6rem;">'
                            f'<div class="kpi-label">Topic {t["id"] + 1}</div>'
                            f'<div style="font-size: 0.82rem; color: #D8D9E0; letter-spacing: 0.05em;">{keywords_pill}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                
                with col_t2:
                    dist = topics_info.get('topic_distribution', {})
                    if dist:
                        dist_df = pd.DataFrame(list(dist.items()), columns=['Topic', 'Count'])
                        fig_topics = px.bar(
                            dist_df, x='Count', y='Topic', orientation='h',
                            color='Count',
                            color_continuous_scale=[[0, '#242730'], [1, '#D4AF37']]
                        )
                        fig_topics.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
                        apply_fashion_layout(fig_topics, "Corpus Distribution by Discovered Theme")
                        st.plotly_chart(fig_topics, use_container_width=True)
            else:
                st.info("Topic decomposition data not available. Execute scripts/process_data.py to compile topics.")

    # 3. TEMPORAL SHIFTS
    with tabs[2]:
        st.markdown("### TEMPORAL TREND SHIFTS")
        st.caption("Evaluation of monthly frequency growth rate to classify trends into Emerging, Rising, Stable, or Declining lifecycle stages.")
        
        shift_df = calculate_trend_shifts(df)
        if not shift_df.empty:
            trends_list = sorted(list(shift_df['detected_trends'].unique()))
            selected_trend = st.selectbox("Select Trend for Lifecycle Inspection", trends_list)
            
            trend_data = shift_df[shift_df['detected_trends'] == selected_trend].copy()
            trend_data['month_str'] = trend_data['month_year'].astype(str)
            
            current_status = trend_data['status'].iloc[-1]
            growth_val = trend_data['growth_rate'].iloc[-1] * 100
            
            cs1, cs2 = st.columns([1, 3])
            with cs1:
                st.markdown(
                    f'<div class="kpi-container">'
                    f'<div class="kpi-label">Lifecycle Stage</div>'
                    f'<div style="margin: 0.4rem 0;"><span class="status-badge">{current_status}</span></div>'
                    f'<div class="kpi-label" style="margin-top: 0.8rem;">Period Growth</div>'
                    f'<div class="kpi-value">{growth_val:+.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            with cs2:
                fig_shift = px.line(
                    trend_data, x='month_str', y='mentions', markers=True,
                    line_shape='spline'
                )
                fig_shift.update_traces(line_color='#D4AF37', marker=dict(size=7, color='#FAF9F6'))
                apply_fashion_layout(fig_shift, f"Monthly Volume Trajectory: {selected_trend.upper()}")
                st.plotly_chart(fig_shift, use_container_width=True)
            
            st.markdown("#### PERIOD-OVER-PERIOD BREAKDOWN")
            table_view = trend_data[['month_str', 'mentions', 'prev_mentions', 'growth_rate', 'status']].copy()
            table_view.columns = ['Month', 'Mentions', 'Prior Period', 'Growth Rate', 'Lifecycle State']
            table_view['Growth Rate'] = table_view['Growth Rate'].apply(lambda g: f"{g * 100:+.1f}%")
            st.dataframe(table_view, use_container_width=True)
        else:
            st.info("Insufficient temporal records to calculate lifecycle transitions.")

    # 4. SENTIMENT INTELLIGENCE
    with tabs[3]:
        st.markdown("### AUDIENCE SENTIMENT DYNAMICS")
        sentiment_counts = df['predicted_sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        c_pie, c_bar = st.columns([1, 1.3])
        with c_pie:
            palette = {'positive': '#7DA88B', 'neutral': '#8E909A', 'negative': '#C57B7B'}
            fig_pie = px.pie(
                sentiment_counts, values='Count', names='Sentiment',
                hole=0.55,
                color='Sentiment',
                color_discrete_map=palette
            )
            apply_fashion_layout(fig_pie, "Global Sentiment Apportionment")
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c_bar:
            exploded = df.explode('detected_trends').dropna(subset=['detected_trends'])
            sent_trend = exploded.groupby(['detected_trends', 'predicted_sentiment']).size().reset_index(name='count')
            top_10 = [t[0] for t in Counter(all_trends).most_common(10)]
            sent_trend_top = sent_trend[sent_trend['detected_trends'].isin(top_10)]
            
            fig_sent_trend = px.bar(
                sent_trend_top, x='detected_trends', y='count', color='predicted_sentiment',
                barmode='stack',
                color_discrete_map=palette
            )
            apply_fashion_layout(fig_sent_trend, "Sentiment Divergence across Top Aesthetics")
            fig_sent_trend.update_layout(xaxis_title="", yaxis_title="Record Count", legend_title="")
            st.plotly_chart(fig_sent_trend, use_container_width=True)

    # 5. INFLUENCER RADAR
    with tabs[4]:
        st.markdown("### INFLUENCER AESTHETIC RADAR")
        influencers = sorted(df['influencer'].unique())
        selected_inf = st.selectbox("Select Creator Profile", influencers)
        
        inf_data = df[df['influencer'] == selected_inf]
        inf_trends = [t for sublist in inf_data['detected_trends'] for t in sublist]
        top_inf_trend = Counter(inf_trends).most_common(1)[0][0] if inf_trends else "None"
        
        ic1, ic2 = st.columns([1, 2.5])
        with ic1:
            st.markdown(
                f'<div class="kpi-container">'
                f'<div class="kpi-label">Creator Handle</div>'
                f'<div style="font-size: 1.1rem; font-weight: 600; color: #FAF9F6; margin-bottom: 0.8rem;">{selected_inf}</div>'
                f'<div class="kpi-label">Monitored Entries</div>'
                f'<div class="kpi-value">{len(inf_data)}</div>'
                f'<div class="kpi-label" style="margin-top: 0.8rem;">Signature Aesthetic</div>'
                f'<div style="font-size: 1.05rem; color: #D4AF37; font-family: Cinzel, serif;">{top_inf_trend.upper()}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            
        with ic2:
            if inf_trends:
                inf_trend_df = pd.DataFrame(Counter(inf_trends).most_common(8), columns=['Trend', 'Mentions'])
                fig_inf = px.bar(
                    inf_trend_df, x='Mentions', y='Trend', orientation='h',
                    color='Mentions',
                    color_continuous_scale=[[0, '#22252C'], [1, '#D4AF37']]
                )
                fig_inf.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
                apply_fashion_layout(fig_inf, f"Aesthetic Portfolio: {selected_inf}")
                st.plotly_chart(fig_inf, use_container_width=True)

    # 6. TEXT ANALYZER
    with tabs[5]:
        st.markdown("### REAL-TIME LINGUISTIC & SENTIMENT INFERENCE")
        st.caption("Submit custom captions or user critiques for immediate pipeline extraction and neural sentiment classification.")

        user_text = st.text_area(
            "Input Social Caption or Feedback",
            "Obsessed with this oversized denim trench coat, beautifully structured and timeless aesthetic.",
            height=100
        )

        fine_tuned_ready = is_fine_tuned_model_available()

        m_col1, m_col2 = st.columns([1.5, 1])
        with m_col1:
            model_choice = st.radio(
                "Classification Engine",
                options=["Fine-Tuned Domain Model (DistilBERT Sequence Classifier)", "Zero-Shot Pre-Trained Transformer"],
                horizontal=True
            )

        if st.button("RUN INFERENCE"):
            with st.spinner("Processing through spaCy and PyTorch architecture..."):
                cleaned = preprocess_text(user_text)
                noun_phrases = extract_noun_phrases(user_text)
                trends = extract_trends_from_phrases([noun_phrases])[0]

                if "Fine-Tuned" in model_choice and fine_tuned_ready:
                    ft_model = load_fine_tuned_model()
                    if ft_model is None:
                        st.error("Failed to load domain-trained model.")
                        st.stop()
                    result = ft_model.predict_single(user_text)
                    sentiment = result["label"]
                    confidence = result["confidence"]
                    scores = result["scores"]
                    engine_label = "Fine-Tuned DistilBERT (Fashion Domain Checkpoint)"
                else:
                    if model is None:
                        st.error("Inference engine unavailable.")
                        st.stop()
                    sentiment = model.predict([user_text])[0]
                    confidence = None
                    scores = None
                    engine_label = "Pre-Trained Transformer (DistilBERT Baseline)"

            st.markdown("---")
            out_c1, out_c2 = st.columns([1.1, 1])
            with out_c1:
                st.markdown("#### LINGUISTIC EXTRACTION (SPACY)")
                st.markdown(
                    f'<div class="kpi-container" style="padding: 1rem 1.2rem;">'
                    f'<div class="kpi-label">Normalized Tokens</div>'
                    f'<div style="font-size: 0.86rem; color: #D8D9E0; margin-bottom: 0.8rem;">{cleaned if cleaned else "No alphanumeric tokens retained"}</div>'
                    f'<div class="kpi-label">Syntactic Noun Chunks</div>'
                    f'<div style="font-size: 0.86rem; color: #D8D9E0; margin-bottom: 0.8rem;">{", ".join(noun_phrases) if noun_phrases else "None extracted"}</div>'
                    f'<div class="kpi-label">Matched Aesthetics / Trends</div>'
                    f'<div style="font-size: 0.86rem; color: #D4AF37; font-weight: 500;">{", ".join(trends) if trends else "No taxonomy entity identified"}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with out_c2:
                st.markdown("#### POLARITY EVALUATION")
                badge_class = {
                    "positive": "sentiment-badge-pos",
                    "negative": "sentiment-badge-neg",
                    "neutral": "sentiment-badge-neu"
                }.get(sentiment.lower(), "sentiment-badge-neu")

                st.markdown(
                    f'<div class="kpi-container" style="padding: 1rem 1.2rem;">'
                    f'<div class="kpi-label">Inference Engine</div>'
                    f'<div style="font-size: 0.8rem; color: #8E909A; margin-bottom: 0.6rem;">{engine_label}</div>'
                    f'<div class="kpi-label">Polarity Verdict</div>'
                    f'<div style="margin: 0.3rem 0 0.8rem 0;"><span class="{badge_class}">{sentiment.upper()}</span></div>'
                    f'{f"<div class=\"kpi-label\">Confidence Level</div><div class=\"kpi-value\" style=\"font-size: 1.4rem;\">{confidence}%</div>" if confidence else ""}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                if scores:
                    score_df = pd.DataFrame([scores]).T.reset_index()
                    score_df.columns = ["Class", "Probability"]
                    fig_score = px.bar(
                        score_df, x="Probability", y="Class", orientation='h',
                        color="Class",
                        color_discrete_map={"positive": "#7DA88B", "neutral": "#8E909A", "negative": "#C57B7B"}
                    )
                    apply_fashion_layout(fig_score, "Posterior Class Probabilities (%)")
                    fig_score.update_layout(showlegend=False, xaxis_title="Percentage", yaxis_title="")
                    st.plotly_chart(fig_score, use_container_width=True)

    # 7. DATASET REPOSITORY
    with tabs[6]:
        st.markdown("### CORPUS DATASET REPOSITORY")
        st.dataframe(df, use_container_width=True)
        
        st.markdown("#### INGEST EXTERNAL CORPUS")
        uploaded_file = st.file_uploader("Upload Delimited CSV for Batch Pipeline Ingestion", type=['csv'])
        if uploaded_file is not None:
            st.info("File successfully staged. Run scripts/process_data.py to execute the NLP taxonomy and inference pipeline on the newly uploaded data.")

    # 8. MODEL PERFORMANCE
    with tabs[7]:
        st.markdown("### EMPIRICAL MODEL PERFORMANCE")
        report_path = "reports/evaluation_report.json"
        cm_path = "reports/confusion_matrix.png"
        
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
                
            c_acc, c_macro = st.columns([1, 3])
            with c_acc:
                st.markdown(
                    f'<div class="kpi-container">'
                    f'<div class="kpi-label">Validation Accuracy</div>'
                    f'<div class="kpi-value">{report["accuracy"] * 100:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
            st.markdown("#### CLASSIFICATION METRICS")
            report_df = pd.DataFrame(report['classification_report']).transpose()
            st.dataframe(report_df.style.format(precision=3), use_container_width=True)
            
            if os.path.exists(cm_path):
                st.markdown("#### CONFUSION MATRIX")
                st.image(cm_path, width=600)
        else:
            st.info("Evaluation metrics report not found. Execute scripts/evaluate_model.py to compile metrics.")

if __name__ == "__main__":
    main()
