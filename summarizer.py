#IMPORTS
from transformers import pipeline
summarizer = pipeline("summarization",model="facebook/bart-large-cnn")
from topic_modelling import get_topics
import re
from load_data import get_sentiment_distribution
import pandas as pd
df = pd.read_csv("training.1600000.processed.noemoticon.csv", names=['sentiment', 'id', 'date', 'query', 'user', 'text'], encoding='latin-1')
negative = df[df['sentiment'] == 0].head(500)
positive = df[df['sentiment'] == 4].head(500)
data = pd.concat([negative, positive])
data = data[['sentiment', 'text']]
data['sentiment'] = data['sentiment'].replace({0: 'negative', 4: 'positive'})

#PRINTING SUMMARY 
'''Takes a long text and outputing a condensed summary of the text. '''
def summarize_insights(text):
 result = summarizer(text)
 summary = result[0]['summary_text']
 return summary 

topics = get_topics()
distribution = get_sentiment_distribution(data)

all_words = []
for topic in topics:
    words = re.findall(r'"(\w+)"', topic[1])
    all_words.extend(words)

insight_text = f"Sentiment distribution: {distribution}. Common topics: {' '.join(all_words)}"

result = summarize_insights(insight_text)
print(result)