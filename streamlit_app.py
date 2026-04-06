import streamlit as st
import pandas as pd
import re
from PyPDF2 import PdfReader
from textblob import TextBlob
import plotly.express as px

st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.title("📊 Product Review Sentiment Analysis")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

def extract_text(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        return text
    except:
        return ""

def split_reviews(text):
    reviews = re.split(r'\n\s*\n', text)
    return [r.strip() for r in reviews if r.strip() and len(r.strip()) > 20]

def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.1:
        return "Positive", score
    elif score < -0.1:
        return "Negative", score
    else:
        return "Neutral", score

if uploaded_file:

    text = extract_text(uploaded_file)

    if not text.strip():
        st.error("❌ No readable text found in PDF (maybe scanned file)")
        st.stop()

    reviews = split_reviews(text)

    if len(reviews) == 0:
        st.warning("⚠️ No valid reviews found")
        st.stop()

    st.success(f"✅ {len(reviews)} reviews extracted")

    data = []
    for i, review in enumerate(reviews):
        label, score = get_sentiment(review)
        data.append({
            "ID": i + 1,
            "Preview": review[:200],
            "Full Review": review,
            "Sentiment": label,
            "Score": round(score, 2)
        })

    df = pd.DataFrame(data)

    total = len(df)
    pos = len(df[df["Sentiment"] == "Positive"])
    neg = len(df[df["Sentiment"] == "Negative"])
    neu = len(df[df["Sentiment"] == "Neutral"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Positive", f"{pos}", f"{(pos/total)*100:.1f}%")
    col2.metric("Negative", f"{neg}", f"{(neg/total)*100:.1f}%")
    col3.metric("Neutral", f"{neu}", f"{(neu/total)*100:.1f}%")

    if not df.empty:

        sentiment_counts = df["Sentiment"].value_counts()

        fig_pie = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            title="Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={
                "Positive": "green",
                "Negative": "red",
                "Neutral": "blue"
            }
        )

        fig_bar = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            title="Sentiment Counts",
            color=sentiment_counts.index,
            color_discrete_map={
                "Positive": "green",
                "Negative": "red",
                "Neutral": "blue"
            }
        )

        fig_hist = px.histogram(
            df,
            x="Score",
            nbins=30,
            title="Polarity Score Distribution"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.plotly_chart(fig_bar, use_container_width=True)

        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("🔍 Filter Reviews")

    selected = st.multiselect(
        "Select Sentiment",
        options=["Positive", "Negative", "Neutral"],
        default=["Positive", "Negative", "Neutral"]
    )

    filtered_df = df[df["Sentiment"].isin(selected)]

    if filtered_df.empty:
        st.warning("⚠️ No data matches selected filter")
    else:
        st.dataframe(filtered_df[["ID", "Preview", "Sentiment", "Score"]])

    st.subheader("📄 Review Details")

    if not df.empty:
        selected_id = st.selectbox("Select Review ID", df["ID"])
        row = df[df["ID"] == selected_id].iloc[0]

        st.info(f"""
        Sentiment: {row['Sentiment']}
        Score: {row['Score']}

        {row['Full Review']}
        """)

    st.subheader("⬇️ Download Results")

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        "Download CSV",
        csv,
        "sentiment_results.csv",
        "text/csv"
    )

else:
    st.info("👆 Upload a PDF file to begin")