# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:11:37 2023

@author: Win 10
"""

# import library
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS

# import data
twitter_data = pd.read_excel('data2archive.xlsx')
twitter_data=twitter_data[['from_user', 'text']]
twitter_data

#Create Wordcloud
texts=str(twitter_data['text'].values).lower()
# Remove the twitter usernames 
texts=re.sub(r'@\w+', ' ', texts)
# Remove the URL
texts=re.sub(r'http\S+', ' ', texts)
# Deleting everything which is not characters
texts = re.sub(r'[^a-z A-Z]', ' ',texts)
# Deleting any word which is less than 3-characters mostly those are stopwords
texts= re.sub(r'\b\w{1,2}\b', '', texts)
# Stripping extra spaces in the text
texts= re.sub(r' +', ' ', texts)
# remove hashtags: texts = re.sub(r'#[A-Za-z0-9_]+', ' ', texts)
texts

#Word cloud
wordcloud_ = WordCloud(
                          background_color='white',
                          stopwords=set(STOPWORDS),
                          max_words=250,
                          max_font_size=30, 
                          random_state=1812
                         ).generate(texts)
def cloud_plot(wordcloud):
    fig = plt.figure(figsize=(10,5))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
cloud_plot(wordcloud_)

#Tweets cleaning function
def clean_text(text):  
    c1 = r'@\w+'                   
    c2 = r'http\S+' 
    c3 = r'[^a-z A-Z]'                      
    c4 = r'\#\w+ '                     
    c5 = r'\b\w{1,2}\b'                                    
    combined_pat = r'|'.join((c1,c2,c3,c4,c5))
    text = re.sub(combined_pat,"",text).lower()
    return text.strip()

twitter_data["cleaned_tweet"] = twitter_data["text"].apply(clean_text)
twitter_data

#Sentiment analysis
polarity = lambda x: TextBlob(x).sentiment.polarity
twitter_data["polarity"] = twitter_data["cleaned_tweet"].apply(polarity)
twitter_data

def sentiment(score):  
    if score == 0:
        return 'Neutral'
    elif score < 0:
        return 'Negative'
    else:
        return 'Positive'
twitter_data['sentiment']=twitter_data["polarity"].apply(sentiment)

#Visualization
twitter_data['sentiment'].value_counts().plot(kind = "bar", color = [(0.8,0.7,0.2),(0.8,0.3,0.2),(0.5,0.8,0.9)])
plt.show()