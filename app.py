import streamlit as st
from sentiment_analysis import pred_text

st.set_page_config(page_title="Sentiment Meme App")
st.title("🧠 Sentiment Meme App")

text = st.text_area("Enter a sentence:")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        sentiment = pred_text(text).strip().lower()
        st.markdown(f"### Sentiment: **{sentiment.capitalize()}**")

        if "positive" in sentiment:
            st.image("positive.png", caption="Positive Meme")
        elif "negative" in sentiment:
            st.image("negative.png", caption="Negative Meme")
        else:
            st.image("neutral.jpg", caption="Neutral Meme")
