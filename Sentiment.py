#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))


# In[42]:


data = pd.read_csv("sentimentdataset.csv")
data.drop(columns = [i for i in data.columns if i not in ["Text","Sentiment"]], inplace = True)


# In[ ]:


def extract_words(sentence):
    cleaned_text = [w.lower() for w in word_tokenize(sentence) if w.lower() not in stop_words]
    return cleaned_text


# In[ ]:


def vocab(corpus):
    vocabulary = []
    
    for doc in corpus:
        words = extract_words(doc)
        vocabulary.extend(words)
        
    vocabulary = sorted(list(set(vocabulary)))
    
    return vocabulary


# In[ ]:


def bow(sentence, vocabulary):
    words = extract_words(sentence)
    bag = np.zeros(len(vocabulary))
    for word in words:
        for i, vocab in enumerate(vocabulary):
            if vocab == word:
                bag[i] += 1
    return bag


# In[ ]:


vocabulary = vocab(data.Text.to_list())


# In[ ]:


arrays = np.empty((0, len(vocabulary)), int)
for val in data.Text.to_list():

    bow_representation = bow(val, vocabulary)
    arrays = np.append(arrays, [bow_representation], axis=0)
    


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['Encoded_Sentiment'] = label_encoder.fit_transform(data['Sentiment'])


# In[ ]:


print("Mapping of original labels to encoded labels:")
for original_label, encoded_label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"{original_label}: {encoded_label}")


# In[ ]:


X = arrays
y = data['Encoded_Sentiment']

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training set
rf_classifier.fit(X, y)

Labels = dict(zip(label_encoder.transform(label_encoder.classes_),label_encoder.classes_))

def pred(text):
    bag = bow(text, vocabulary)
    input_array = np.array([bag])
    y_pred = rf_classifier.predict(input_array)
    return y_pred

# inputt = input("Enter the Text input: ")
# predicted_label = pred(inputt)[0]
# print("Predicted Label:", Labels[predicted_label])


# In[ ]:


import streamlit as st


# In[43]:
emojis = {
    ' Acceptance   ': '😊',
    ' Acceptance      ': '😊',
    ' Accomplishment ': '🏆',
    ' Admiration ': '😍',
    ' Admiration   ': '😍',
    ' Admiration    ': '😍',
    ' Adoration    ': '😍',
    ' Adrenaline     ': '🤯',
    ' Adventure ': '🚀',
    ' Affection    ': '❤️',
    ' Amazement ': '😲',
    ' Ambivalence ': '😐',
    ' Ambivalence     ': '😐',
    ' Amusement    ': '😄',
    ' Amusement     ': '😄',
    ' Anger        ': '😡',
    ' Anticipation ': '😬',
    ' Anticipation  ': '😬',
    ' Anxiety   ': '😰',
    ' Anxiety         ': '😰',
    ' Appreciation  ': '🙏',
    ' Apprehensive ': '😟',
    ' Arousal       ': '😏',
    ' ArtisticBurst ': '🎨',
    ' Awe ': '😲',
    ' Awe    ': '😲',
    ' Awe          ': '😲',
    ' Awe           ': '😲',
    ' Bad ': '😞',
    ' Betrayal ': '😔',
    ' Betrayal      ': '😔',
    ' Bitter       ': '😖',
    ' Bitterness ': '😖',
    ' Bittersweet ': '😔😊',
    ' Blessed       ': '🙏',
    ' Boredom ': '😑',
    ' Boredom         ': '😑',
    ' Breakthrough ': '🎉',
    ' Calmness     ': '😌',
    ' Calmness      ': '😌',
    ' Captivation ': '😍',
    ' Celebration ': '🎉',
    ' Celestial Wonder ': '🌟',
    ' Challenge ': '💪',
    ' Charm ': '😊',
    ' Colorful ': '🎨',
    ' Compassion': '❤️',
    ' Compassion    ': '❤️',
    ' Compassionate ': '❤️',
    ' Confidence    ': '😎',
    ' Confident ': '😎',
    ' Confusion ': '😕',
    ' Confusion    ': '😕',
    ' Confusion       ': '😕',
    ' Connection ': '💑',
    ' Contemplation ': '🤔',
    ' Contentment ': '😊',
    ' Contentment   ': '😊',
    ' Coziness     ': '🏠',
    ' Creative Inspiration ': '🎨',
    ' Creativity ': '🎨',
    ' Creativity   ': '🎨',
    ' Culinary Adventure ': '🍽️',
    ' CulinaryOdyssey ': '🍽️',
    ' Curiosity ': '🤔',
    ' Curiosity  ': '🤔',
    ' Curiosity   ': '🤔',
    ' Curiosity     ': '🤔',
    ' Curiosity       ': '🤔',
    ' Darkness     ': '🌑',
    ' Dazzle        ': '✨',
    ' Desolation ': '😞',
    ' Despair ': '😢',
    ' Despair   ': '😢',
    ' Despair      ': '😢',
    ' Despair         ': '😢',
    ' Desperation ': '😟',
    ' Determination ': '💪',
    ' Determination   ': '💪',
    ' Devastated ': '😭',
    ' Disappointed ': '😞',
    ' Disappointment ': '😞',
    ' Disgust ': '🤢',
    ' Disgust      ': '🤢',
    ' Disgust         ': '🤢',
    ' Dismissive ': '😒',
    ' DreamChaser   ': '🚀',
    ' Ecstasy ': '😆',
    ' Elation   ': '😆',
    ' Elation       ': '😆',
    ' Elegance ': '💃',
    ' Embarrassed ': '😳',
    ' Emotion ': '😄',
    ' EmotionalStorm ': '😱',
    ' Empathetic ': '❤️',
    ' Empowerment   ': '💪',
    ' Enchantment ': '😍',
    ' Enchantment   ': '😍',
    ' Energy ': '⚡',
    ' Engagement ': '💍',
    ' Enjoyment    ': '😊',
    ' Enthusiasm ': '😃',
    ' Enthusiasm    ': '😃',
    ' Envious ': '😠',
    ' Envisioning History ': '🏰',
    ' Envy            ': '😠',
    ' Euphoria ': '😁',
    ' Euphoria   ': '😁',
    ' Euphoria     ': '😁',
    ' Euphoria      ': '😁',
    ' Excitement ': '😃',
    ' Excitement   ': '😃',
    ' Excitement    ': '😃',
    ' Exhaustion ': '😩',
    ' Exploration ': '🌍',
    ' Fear         ': '😨',
    ' Fearful ': '😨',
    ' FestiveJoy    ': '🎉',
    ' Free-spirited ': '🦋',
    ' Freedom       ': '🗽',
    ' Friendship ': '👫',
    ' Frustrated ': '😤',
    ' Frustration ': '😤',
    ' Frustration     ': '😤',
    ' Fulfillment  ': '😊',
    ' Fulfillment   ': '😊'}

def main():
    st.title("Text Sentiment Classifier")
    selected_text = st.selectbox("Select Text", data['Text'].tolist())
    if st.button("Predict"):
        predicted_label = pred(selected_text)[0]
        try:
            st.write("Predicted Sentiment:", Labels[predicted_label], emojis[Labels[predicted_label]])
        except:
            st.write("Predicted Sentiment:", Labels[predicted_label])
        

if __name__ == "__main__":
    main()


# In[ ]:





