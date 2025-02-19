
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
import joblib

import streamlit as st
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")



from app_class import Tokenizer

@st.cache_resource
def load_model():
    logged_model = joblib.load('/app_stream/models/best_model.pkl')
    return logged_model

model = load_model()
tokenizer = Tokenizer()


def predict_sentiment(text):
    x_test_tokenized = tokenizer.preprocess_texts([text], 1)
    x_test_joined = [" ".join(token) for token in x_test_tokenized]
    prediction_proba = model.predict_proba(x_test_joined)[0]
      
    sentiment = "Positive" if prediction_proba[1] > 0.5 else "Negative"
    confidence = max(prediction_proba) 
    return sentiment, confidence, prediction_proba[1]



st.title("📊 Sentiment Analyzer")

st.write("Ingrese un texto y analizaremos su sentimiento.")

user_input = st.text_area("📝 Escribe aquí:", "")


if st.button("🔍 Analizar Sentimiento"):
    if user_input.strip():
       
        sentiment, confidence, proba_pos = predict_sentiment(user_input)

        
        st.subheader(f"Resultado: {sentiment} ({confidence:.2%} confidence)")

    
        st.progress(int(proba_pos * 100))

       
        if sentiment == "Positive":
            st.success("🙂 ¡El texto tiene un sentimiento positivo!")
        else:
            st.error("🙁 El texto tiene un sentimiento negativo.")
    else:
        st.warning("⚠️ Por favor, ingrese un texto antes de analizar.")