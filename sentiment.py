import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from preprocessing import clean_tweet

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(tweet):
    scores = sia.polarity_scores(tweet)
    if scores['compound'] < -0.05:
        return "negative"
    elif -0.05 <= scores['compound'] <= 0.05:
        return "neutral"
    else:
        return "positive"

if __name__ == "__main__":
    tweets = ["I hate this", "Tesla is amazing", "The weather is okay"]
    for tweet in tweets:
        cleaned = " ".join(clean_tweet(tweet))
        sentiment = analyze_sentiment(cleaned)
        print(f"Tweet: {tweet}")
        print(f"Sentiment: {sentiment}")
 
