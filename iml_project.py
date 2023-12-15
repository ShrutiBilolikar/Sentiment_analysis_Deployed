
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.feature_extraction.text import CountVectorizer


bagging_model = pickle.load(open('MLmodel.sav', 'rb'))

vect = pickle.load(open('vect.pkl', 'rb'))  


with st.sidebar:
    
    selected = option_menu('Sentiment analysis',                          
                          ['Sentiment Prediction'],                        
                          default_index=0)
    
    
if (selected == 'Sentiment Prediction'):
    
    
    st.title('Sentiment analysis using machine learning')
    tweet = st.text_input('Text')
    tweet_new = vect.transform([tweet])
    
    pred = ''
    
    
    if st.button('Sentiment analysis result'):
        prediction = bagging_model.predict(tweet_new)
        
        if (prediction[0] == 1):
          pred = 'Positive'
        else:
          pred = 'Negative'
        
    st.success(pred)


