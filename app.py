import streamlit as st
from joblib import load
from nltk.corpus import stopwords
import string

st.title('Fake Review Detection')
st.write('Enter your comment below:')

# Text input for user's comment
user_input = st.text_input('Input your comment here:', '')

def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

if st.button('Predict'):
    try:
        # Attempt to load the model
        model = load('text_classification_model_SVC.joblib')
        model2= load('text_classification_model_rating_SVC.joblib')
        # Make predictions
        prediction = model.predict([user_input])
        rating=model2.predict([user_input])
        
        # Display prediction
        st.write(f'The comment is classified as: {prediction[0]} and rating as : {rating[0]}')
    except Exception as e:
        st.write("An error occurred:", e)

st.subheader("Information on the Models")
if st.checkbox("Performance of various ML models:"):
    
    st.write('1.K Nearest Neighbors Prediction Accuracy: 57.52%')   
    st.write('2.Support Vector Machines Prediction Accuracy: 87.9%')



