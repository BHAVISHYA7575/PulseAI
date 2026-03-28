#IMPORTS
from transformers import pipeline
summarizer = pipeline("summarization",model="facebook/bart-large-cnn")
from topic_modelling import get_topics
import re


#PRINTING SUMMARY 
'''Takes a long text and outputing a condensed summary of the text. '''
def summarize_insights(text):
 result = summarizer(text)
 summary = result[0]['summary_text']
 return summary 


topics = get_topics()
all_words = []
for topic in topics:
   words = re.findall(r'"(\w+)"',topic[1])
   all_words.extend(words)

insight_text = " ".join(all_words)


result = summarize_insights(insight_text)
print(result)