import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords

import requests
from bs4 import BeautifulSoup

import torch
from transformers import BartForConditionalGeneration,BartTokenizer

model_name = 'Facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def summarize_long_text(input_text, max_chunk_length=500, max_summary_length=150):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Split the input text into chunks
    chunks = [input_text[i:i+max_chunk_length] for i in range(0, len(input_text), max_chunk_length)]

    # Generate summaries for each chunk
    summaries = []
    for chunk in chunks:
        input_ids = tokenizer.encode(chunk, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(input_ids, max_length=max_summary_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary_text)

    # Combine the summaries of all chunks
    final_summary = " ".join(summaries)
    return final_summary

def get_text_from_website(url):
    # Make a request to the website and get the HTML content
    response = requests.get(url)
    html_content = response.text

    # Use BeautifulSoup to extract text from HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    text = ' '.join([p.get_text() for p in soup.find_all('p')])

    return text

def main():
    while True:
        print("Choose an option:")
        print("1. Enter text manually")
        print("2. Summarize content from a website")
        print("3. End the task")

        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == "1":
            print("Enter the text you want to summarize (press Enter to finish):")
            user_input = []
            while True:
                line = input()
                if not line:
                    break
                user_input.append(line)

            input_text = " ".join(user_input)
        elif choice == "2":
            website_url = input("Enter the URL of the website you want to summarize: ").strip()
            input_text = get_text_from_website(website_url)
        elif choice == "3":
            print("Ending the task. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
            continue

        summary = summarize_long_text(input_text)

        print("\nOriginal Text:")
        print(input_text[:500])  # Print the first 500 characters for brevity
        print("\nSummary:")
        print(summary)

if __name__ == "__main__":
    main()
