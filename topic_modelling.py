#IMPORTS
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# TOPIC MODELLING 
''' Topic modelling is a technique used in NLP to discover hidden topics in documents.'''

#ESSENTIAL LIBRARIES
import pandas as pd
from preprocessing import clean_tweet 

#DATASET
df = pd.read_csv("training.1600000.processed.noemoticon.csv", names=['sentiment', 'id', 'date', 'query', 'user', 'text'], encoding = 'latin-1')

data = df.head(1000)  # getting 1000rows 
data = data[['text']] # getting only the text column 
print(data) # printing that data 

#APPLYING TWEET TRANSOFRMATION 
cleaned_tweet = data['text'].apply(clean_tweet) # on the text data we are applying clean tweet function
''' so data text is a column of 1000 tweets that only contain text and that text is raw means unprocessed
  we apply clean tweet we pass it through a pipeline 
  which transform the text and give us clean words.
  pipeline includes lowercase, no links, no mentions, stopwords removed, lemmatized.
  now we see 1000 tweets clean and can be used for further transformation '''
print(cleaned_tweet)

#BUILDING DICTIONARY 
dictionary = Dictionary(cleaned_tweet)
print(f"Dictionary size: {len(dictionary)}")

#CORPUS 
corpus = [dictionary.doc2bow(tweet) for tweet in cleaned_tweet]
print (corpus[0])
print(dictionary[0])

#IMPLEMENTING LDA MODEL 
lda_model = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary)
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)


