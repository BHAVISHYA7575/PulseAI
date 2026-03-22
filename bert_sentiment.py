from transformers import pipeline 

sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

tweet = "Elon Musk announced Tesla will open a factory in Berlin!!! @elonmusk https://t.co/xyz123 "

label_map = {'LABEL_0': "negative", 'LABEL_1' : "neutral", 'LABEL_2' : "positive"}

def bert_analyze_sentiment(tweet):
    result = sentiment_pipeline(tweet)
    label = result[0]['label']
    result[0]['label'] 
    converted = label_map[label]
    return converted 

result = bert_analyze_sentiment(tweet)
print(result)
