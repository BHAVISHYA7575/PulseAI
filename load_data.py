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