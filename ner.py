import re 
import spacy 
nlp = spacy.load("en_core_web_sm")

tweet = "Elon Musk announced Tesla will open a factory in Berlin!!! @elonmusk https://t.co/xyz123 🚀"

def clean_for_ner(tweet):
#REMOVING LINKS 
 tweet = re.sub(r'http\S+', '', tweet)
#REMOVING MENTIONS 
 tweet = re.sub(r'@\S+', '', tweet)
# REMOVING PUNCTUTUATION
 tweet = re.sub(r'[^a-zA-Z0-9\s]', '', tweet)

 return tweet

def extract_entities(tweet):
    doc = nlp(tweet)
    entity = []
    for ents in doc.ents:
        entity.append((f"Entity: {ents.text}, Label: {ents.label_}"))
       
    return entity

if __name__ == "__main__":
 clean_tweet =clean_for_ner(tweet)
 extract = extract_entities(clean_tweet) 
 print(extract)