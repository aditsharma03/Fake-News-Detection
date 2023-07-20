import streamlit as st
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import PassiveAggressiveClassifier
import sklearn
import pickle

# Set the page title and layout
st.title("News Article Input")
st.write("Enter your news article below and click the submit button to check its reliability.")

# Create a text box for the user to input the news article
article = st.text_area("News Article Here:", height=300)

# Create a submit button
if st.button("Submit"):
    # Store the entire text in the 'str' variable
    content = article

    # Check if the variable is not empty
    if content:
        # Perform operations if the variable is not empty
        tfidf = pickle.load(open('TfidfVectorizer.sav', 'rb'))
        pac = pickle.load(open('PassiveAggressiveClassifier.sav', 'rb'))

        # Normalizing the article
        content = tfidf.transform([content])

        label_list = pac.predict(content)
        is_reliable = label_list[0] == 1

        # Display a green box indicating the news is reliable
        if is_reliable:
            st.success("The news is not fake and is reliable.")
        else:
            st.error("The news is fake and not reliable.")
    else:
        # Perform operations if the variable is empty
        st.warning("Please enter a news article to check its reliability.")
