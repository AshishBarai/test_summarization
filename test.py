import streamlit as st
from summarization_functions import summarize_long_text, get_text_from_website

def main():
    st.title("Text Summarization App")

    option = st.sidebar.selectbox("Choose an option:", ["Enter text manually", "Summarize content from a website"])

    if option == "Enter text manually":
        user_input = st.text_area("Enter the text you want to summarize:")
        if st.button("Summarize"):
            summary = summarize_long_text(user_input)
            st.subheader("Summary:")
            st.write(summary)

    elif option == "Summarize content from a website":
        website_url = st.text_input("Enter the URL of the website you want to summarize:")
        if st.button("Summarize"):
            input_text = get_text_from_website(website_url)
            summary = summarize_long_text(input_text)
            st.subheader("Summary:")
            st.write(summary)

if __name__ == "__main__":
    main()
