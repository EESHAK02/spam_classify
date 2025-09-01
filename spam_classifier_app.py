import streamlit as st
import joblib

model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("Spam SMS Classifier")
st.write("Enter a message to check if it is spam or not")

user_in = st.text_area("Your Message:")

if st.button("Predict"):
    if user_in.strip() == "":
        st.warning("Please enter a valid message!")
    else:
        inp_vec = vectorizer.transform([user_in])
        pred = model.predict(inp_vec)[0]

        if pred == 1:
            st.error("This is a SPAM message!")
        else:
            st.success("This message is safe!")