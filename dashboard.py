#IMPORTS
import streamlit as st
from load_data import get_sentiment_distribution
from topic_modelling import get_topics
from summarizer import summarize_insights
import pandas as pd 
import re

#TITLE OF THE DASHBOARD
st.title("PULSEAI Dashboard - Social Media Dashboard")
#SUBTITLE OF THE DASHBOARD
st.subheader("Real time Twitter/X Sentiment & Topic Analysis")

#DATA LOADING 
df = pd.read_csv("training.1600000.processed.noemoticon.csv", names=['sentiment', 'id', 'date', 'query', 'user', 'text'], encoding='latin-1')
negative = df[df['sentiment'] == 0].head(500)
positive = df[df['sentiment'] == 4].head(500)
data = pd.concat([negative, positive])
data = data[['sentiment', 'text']]
data['sentiment'] = data['sentiment'].replace({0: 'negative', 4: 'positive'})

@st.cache_data
def load_distribution(_data):
    return get_sentiment_distribution(_data)

@st.cache_data  
def load_topics():
    return get_topics()

#FUNCTION CALLING WITH SPINNER 
with st.spinner("Analyzing 1000 tweets...this may takes a few minutes"):
 distribution = load_distribution(data)
 topics = load_topics()

#DISPLAYING THE SENTIMENT DISTRIBUTION
st.divider()
st.subheader("Sentiment Distribution")
col1, col2, col3 = st.columns(3)
col1.metric("Positive", distribution['positive'])
col2.metric("Negative", distribution['negative'])
col3.metric("Neutral", distribution['neutral'])
st.bar_chart(distribution)

#DISPLAYING THE TOPICS
st.divider()
st.subheader("Discovered Topics")
for topic in topics:
    st.write(topic)

#DISPLAYING THE INSIGHTS SUMMARY
st.divider()
st.subheader("Insights Summary")
all_words = []
for topic in topics:
    words = re.findall(r'"(\w+)"', topic[1])
    st.write(f"Topic {topic[0]}: {' '.join(words)}")    
    all_words.extend(words)
insight_text = f"Sentiment distribution: {distribution}. Common topics: {' '.join(all_words)}"
summary = summarize_insights(insight_text)
st.write(summary)
