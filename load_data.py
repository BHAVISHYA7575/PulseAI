import pandas as pd
from bert_sentiment import bert_analyze_sentiment 


correct = 0 
total = 0 

df = pd.read_csv("training.1600000.processed.noemoticon.csv", names=['sentiment', 'id', 'date', 'query', 'user', 'text'], encoding = 'latin-1')

negative = df[df['sentiment'] == 0].head(500)
positive = df[df['sentiment'] == 4].head(500)
data = pd.concat([negative,positive])
print(data)

data = data[['sentiment', 'text']]
print(data.head())

data['sentiment'] = data['sentiment'].replace({0: 'negative', 4: 'positive'})
print(data.head())

for index, row in data.iterrows():
 tweet = row['text']
 sentiment = bert_analyze_sentiment(tweet)
 
 print(f"Actual: {row['sentiment']}, Predicted: {sentiment}")
 total = total + 1 
 if (row['sentiment'] == sentiment):
  correct = correct+1
print(f"Correct: {correct}")
print(f"Total: {total}")
print(f"Accuracy: {correct/total * 100}%")

#DISTRIBUTING THE SENTIMENTS IN THE DATASET
def get_sentiment_distribution(data):
  positive_count = 0
  negative_count = 0
  neutral_count = 0
  for index, row in data.iterrows():
    tweet = row['text']
    sentiment = bert_analyze_sentiment(tweet)
    if sentiment == "positive":
      positive_count += 1
    elif sentiment == "negative":
      negative_count += 1
    else :
      neutral_count += 1
  return {"positive": positive_count, "negative": negative_count, "neutral": neutral_count}

if __name__ == "__main__":
    distribution = get_sentiment_distribution(data)
    print(distribution)