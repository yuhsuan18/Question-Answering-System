import io
import random
import string 
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
import requests
import json
import re
import time

# Downloading and installing NLTK 
#pip install nltk

# for downloading packages
#nltk.download('popular', quiet=True) 
#nltk.download('punkt') 
#nltk.download('wordnet') 

import nltk
from nltk.stem import WordNetLemmatizer


# Collect keywords
intent_dict = {'1':'weather','2':'food','3':'traffic','4':'greetings'}

weather_dict = ['weather','sunny','windy','rainy','rain','snow','fogggy','forecast','cold','warm','cool','hot','temp','temperature']
food_dict = ['restaurant','food','breakfast','lunch','dinner','eat','brunch','meal','coffee','fika','cafe','wine','dine','hungry','starve','go','date','romantic','book','table','service']
transport_dict = ['bus','tram','transport','wait','Vagnhallen Majorna','Centralstationen','next','time']

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey","hej", "hej hej hej")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "what's up","hej", "hej hej hej"]


# ### Functions for weather

def getweather(city):

    request = "http://api.openweathermap.org/data/2.5/forecast?q=" + city + "&units=metric&cnt=40&appid=d910cdfca2b02ed3e3922ac6e61e1463"
    response = requests.get(request)

    def jprint(obj):
        text = json.dumps(obj,sort_keys=True,indent=4)
        with open("output.json","w") as outfile:
            outfile.write(text)

    jprint(response.json())
    
    f = open ('output.json', "r") 
    data = json.loads(f.read())
        
    with open('forecast.txt','w') as fp:
        for line in data['list']:
            fp.write(line['dt_txt'])
            fp.write('\n')
            
            for w in line['main'].items():   
                fp.writelines(str(w))
                fp.write('\n')

            for words in line['weather']: 
                fp.writelines(str(words))

            fp.write('\n')
            fp.write('\n')


def datetoweather(date,time='12:00'):

    date_string = '2021' + '-' + date[0:2] + '-' + date[2:4]
    time_string = time + ':00'
    
    return tuple([date_string,time_string])  # e.g. (2021-03-08 ,15:00:00)
    

def process_weather(user_response,responseidx):
    word_dict ={}
    with open('forecast.txt','r') as fp:
        for k in range(40):
            key = fp.readline()
            key = key.replace('\n','')
            key  = tuple(key.split())
            words_list = []
            for i in range(10):
                line = fp.readline()
                line = line.replace('\n','')
                words_list.append(line)
            fp.readline()
            word_dict[key] = words_list


    response_list = user_response.lower().split()


    if key in word_dict.keys():
        
        if 'highest' in response_list or 'max' in response_list:
            respon = word_dict[responseidx][7]
            respon = re.sub('[,\(\)\']','',respon)
            print("BOBO: ",respon + " deg. Celcius")  
        elif 'lowest' in response_list or 'min' in response_list:
            respon = word_dict[responseidx][8]
            respon = re.sub('[,\(\)\']','',respon)
            print("BOBO: ",respon + " deg. Celcius")
        elif 'temperature' in response_list or 'temp' in response_list or 'hot' in response_list or 'cold' in response_list:
            respon = word_dict[responseidx][0]
            respon = re.sub('[,\(\)\']','',respon)
            print("BOBO: ",respon + " deg. Celcius")
        elif 'rain' in response_list:
            respon = word_dict[responseidx][2]
            respon = re.sub('[,\(\)\']','',respon)
            print("BOBO: ",respon + " %")


# ### Functions for traffic

def process_traffic(user_response):
    tram_list = []
    with open('traffic.txt','r') as fp:
        for i in range(10):
            fp.readline()
            tram_no = fp.readline()
            tram_no = tram_no.replace('\n','')
            departure = fp.readline()
            departure = departure.replace('\n','')
            hour = departure.split()[1][5:7]
            minute = departure.split()[1][8:10]
            duration = fp.readline()
            duration = duration.replace('\n','')
            duration = duration.split()[2]
            tram_list.append((tram_no,hour,minute,duration))
            fp.readline()
        # print(tram_list)
        
    t = time.localtime()
    current_time = time.strftime("%H:%M", t)
    
    current_hr = current_time[:2]
    current_minute = current_time[3:]
    
    for item in tram_list:
        count = 0
        if current_hr > item[1]:
            continue
        elif current_hr < item[1]:
            count +=1
            print("next tram/bus: ", item[0])
            print("next departure time: ",item[1]+':'+item[2])
        elif current_hr == item[1] and current_minute > item[2]:
            continue
        else:
            count +=1
            print("next tram/bus: ", item[0])
            print("next departure time: ",item[1]+':'+item[2])
    if count == 0:
        print("sorry! No tram or bus")
                
            


# ### General functions
def process_intent(sentence):

    intent = -1
    sentence = sentence.lower()
    translation = sentence.maketrans(string.punctuation,string.punctuation,string.punctuation) # remove punctuation
    sentence = sentence.translate(translation)
    for word in sentence.split():
        if word in weather_dict:
            intent = 1
        elif word in food_dict:
            intent = 2
        elif word in transport_dict:
            intent = 3
        elif word in GREETING_RESPONSES:
            intent = 4

    return int(intent)


#Reading in the corpus
def read_corpus(intent):
    
    if intent == 1:    
        f=open('forecast.txt','r',errors = 'ignore',encoding='utf-8')
        raw=f.read()
        raw = raw.lower()# converts to lowercase
    elif intent == 2:
        f=open('restaurant_eng.txt','r',errors = 'ignore',encoding='utf-8')
        raw=f.read()
        raw = raw.lower()# converts to lowercase
    elif intent == 3:
        f=open('traffic.txt','r',errors = 'ignore',encoding='utf-8')
        raw=f.read()
        raw = raw.lower()# converts to lowercase
    else:
        print(" Sorry!! services are not support right now")        
        #Tokenisation
    sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
    word_tokens = nltk.word_tokenize(raw)# converts to list of words    
    return  sent_tokens, word_tokens


# Preprocessing
#WordNet is a semantically-oriented dictionary of English included in NLTK

def LemTokens(tokens):
    lemmer = nltk.stem.WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


#Keyword matching
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        
def response(user_response, intent):
    robo_response=''
    
    
    if (intent == 1):  # weather
        
        city = input("BOBO: which city do you want to know? ")
        getweather(city)
        
        
        date = input("BOBO: which date? (MMDD) ")
        time = input("BOBO: what time? (06:00 / 12:00 / 18:00) ")
        
        responseidx = datetoweather(date,time)
        process_weather(user_response,responseidx)
        
        return robo_response
                   
    elif (intent == 2): # food
        pass
    elif (intent == 3): # traffic
        process_traffic(user_response)
        
        return robo_response
        
    else: # greeting
        pass
    
    
    sent_tokens, word_tokens = read_corpus(intent)
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you. Can you give more details ?"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response
    

def get_response(user_response):
    
    
    flag = True
    
    intent = process_intent(user_response)
    
    
    if intent == -1:
        print("We are sorry, it's not support now !")
        return
    
    if intent == 4:
        print("BOBO: "+ greeting(user_response))
    else:
        print("BOBO: My name is Bobo. I will answer your queries. If you want to exit, type Bye!\n")
        print("BOBO: Welcome to ",intent_dict[str(intent)], "Section")
    
    
    while(flag==True):

        
        user_response=user_response.lower()
        if(user_response!='bye'):
            
            if(user_response =='thanks' or user_response =='thank you' or user_response == 'tack'):
                flag=False
                print("BOBO: You are welcome..")
            else:
                if intent != 4:

                    print(response(user_response, intent))
                
                   
        else:
            flag=False
            print("BOBO: Bye! take care..")
            return
            
            
        user_response = input("BOBO: type bye to exit or type something to carry on\n ")
        if user_response =='bye':
            print("BOBO: see u next time !! ")
            break
        
        intent = process_intent(user_response)
        
        
        
        if intent == -1:
            print("BOBO: We are sorry, it's not support now !\n")
            return
        
        if intent == 4:
            print("BOBO: "+ greeting(user_response))
            

def main():
    user_response = input("Hej hej, what do you want to know?\n")

    get_response(user_response)


# In[ ]:


if __name__ == "__main__":
    
    main()


