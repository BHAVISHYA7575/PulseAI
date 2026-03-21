from preprocessing import clean_tweet
from sentiment import analyze_sentiment
from ner import extract_entities
from ner import clean_for_ner 
tweet = "Elon Musk announced that Tesla will open a new factory in Berlin"

cleaned_tweet = (clean_tweet(tweet))
cleaned_tweet = " ".join(cleaned_tweet)
sentiment = analyze_sentiment(cleaned_tweet)
ner_cleaned = clean_for_ner(tweet)
entities = extract_entities(ner_cleaned)

print(f"Cleaned tweet: {cleaned_tweet}")
print(f"Sentiment : {sentiment}")
print(f"Entities: {entities}")