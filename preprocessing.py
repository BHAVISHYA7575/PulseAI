import re 
import nltk
nltk.download('stopwords')
from nltk.corpus  import stopwords
stop_words = set(stopwords.words('english'))
custom_stopwords = {"im", "u", "ur", "lol", "omg", "rt", "ha", "wa", "gonna", "wanna", "gotta"}
stop_words.update(custom_stopwords)
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

tweet = "Tesla stock is AMAZING!!! @elonmusk https://t.co/xyz123 🚀🚀 It's the best... #Tesla #Stocks"

lemmatizer = WordNetLemmatizer()

#FUNCTION CREATION FOR CLEANING THE TWEET 
def clean_tweet(tweet):
 
# LOWERCASING 
 tweet = tweet.lower()

#REMOVING LINKS 
 tweet = re.sub(r'http\S+', '', tweet)

#REMOVING MENTIONS 
 tweet = re.sub(r'@\S+', '', tweet)

# REMOVING PUNCTUTUATION
 tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)

#REMOVING STOPWORDS 
 tweet = tweet.split()
 clean_line = [] 
 for word in tweet: 
  if word not in stop_words: 
   word = lemmatizer.lemmatize(word)
   clean_line.append(word) 

 return clean_line 
if __name__ == "__main__":
    result = clean_tweet(tweet)
    print(result)



