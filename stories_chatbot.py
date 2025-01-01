import random
import json
import pickle
import numpy as np
import nltk
import tensorflow as tf

from nltk.stem import WordNetLemmatizer

#from keras.models import load_model
#from tensorflow.python.keras.models import load_model

lemmatizer = WordNetLemmatizer()
#intents = json.loads(open('C:\Users\Raj\Desktop\ML_Internship\Stories_Chatbot\intents.json').read())
#intents = json.loads(open("stories_intents.json").read())

# Based on Latest Updated Data
intents = json.loads(open("updated_intents.json").read())

words = pickle.load(open('updated_intents_words.pkl', 'rb'))
classes = pickle.load(open('updated_intents_classes.pkl', 'rb'))

#words = pickle.load(open('stories_words.pkl', 'rb'))
#classes = pickle.load(open('stories_classes.pkl', 'rb'))

#model = load_model('chatbot_model.h5')     // This is Original Command but it's Not Working.

#model = tf.keras.models.load_model('stories_chatbot_model.h5')
# Based on Latest Updated Data
model = tf.keras.models.load_model('updated_stories_chatbot_model.h5')

#model = tf.keras.layers.TFSMLayer("chatbot_model", call_endpoint="serving_default")



def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class (sentence):
    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break
    return result

#print("GO! Bot is running!")
print("Hello from Stories.")

while True:
    message = input("")
    ints = predict_class (message)
    res = get_response (ints, intents)
    print (res)
   