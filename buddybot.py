import numpy as np
import nltk 
nltk.download('punkt_tab')
import string
import random

f=open('Sample.txt','r',errors='ignore')
raw_doc=f.read()

raw_doc = raw_doc.lower() #converting entire text to lowercase
nltk.download('punkt') #using the punkt tokenizer
nltk.download('wordnet') #using the wordnet dictionary
nltk.download('omw-1.4')
sentence_tokens=nltk.sent_tokenize(raw_doc)
word_tokens=nltk.word_tokenize(raw_doc)
sentence_tokens[:5]
word_tokens[:5]

lemmer=nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punc_dict= dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

greet_inputs=("hello","hi","hey","hey you", "what's up?","how are you?", "howdy!")
greet_responses=("hi!","hey!","hi there!", "hey there!","nice to meet you!", "pleased to meet you!", "hello, long time no see!")
def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    robo1_response=""
    TfidfVec= TfidfVectorizer(tokenizer= LemNormalize, stop_words="english")
    tfidf= TfidfVec.fit_transform(sentence_tokens)
    vals= cosine_similarity(tfidf[-1], tfidf)
    idx= vals.argsort()[0][-2]
    flat= vals.flatten()
    flat.sort()
    req_tfidf= flat[-2]
    if(req_tfidf==0):
        robo1_response= robo1_response+"I am sorry. Unable to understand you!"
        return robo1_response
    else:
        robo1_response= robo1_response+ sentence_tokens[idx]
        return robo1_response


user_response=input("User: ")
flag= True
print("Hello! I am BuddyBot.\
\nStuck in a sea of text? Need to quickly understand a document but short on time?\
\nBuddyBot is here to help you! Just feed it any document, and it will become your on-demand information source.\
\nAsk BuddyBot anything about the content - key points, specific details.\
\nBuddyBot will analyze the text and provide clear, concise answers, saving you valuable time and effort.\
\nUnleash the power of your documents with BuddyBot!\
\nFor ending conversation, type bye!")

while(flag==True):
    user_response=input("User: ")
    user_response= user_response.lower()
    if(user_response!="bye"):
        if(user_response=="thank you" or user_response=="thanks"):
            flag=False
            print("BuddyBot: You are welcome..")
        else:
            if(greet(user_response)!=None):
                print("BuddyBot: "+greet(user_response))
            else:
                sentence_tokens.append(user_response)
                word_tokens=word_tokens+ nltk.word_tokenize(user_response)
                final_words=list(set(word_tokens))
                print("BuddyBot: ", end="")
                print(response(user_response))
                sentence_tokens.remove(user_response)
    else:
        flag=False
        print("BuddyBot: Goodbye!")

