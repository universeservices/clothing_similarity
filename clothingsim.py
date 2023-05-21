import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json

# Step 2: Scraping the clothing descriptions
def scrape_clothing_descriptions(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    description_elements = soup.find_all('span', class_='a-size-base-plus a-color-base a-text-normal')
    descriptions = [element.get_text() for element in description_elements]
    return descriptions

# Step 3: Cleaning and preprocessing the data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Step 4: Feature extraction
def extract_features(descriptions):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(descriptions)
    return features

# Step 5: Measuring similarity
def calculate_similarity(features):
    similarity_matrix = cosine_similarity(features)
    return similarity_matrix

# Step 6: Ranking the similar items
def rank_similar_items(similarity_matrix, item_index):
    similarity_scores = similarity_matrix[item_index]
    sorted_indices = similarity_scores.argsort()[::-1]
    ranked_items = [index for index in sorted_indices if index != item_index]
    return ranked_items

# Main code
def clothing_similarity_search(request):
    # Step 2: Scraping clothing descriptions from Amazon.com
    url = "https://www.amazon.com/s?k=clothing"
    print("Step 2: Scraping clothing descriptions...")
    descriptions = scrape_clothing_descriptions(url)
    print("Scraped", len(descriptions), "clothing descriptions")

    # Step 3: Clean and preprocess the data
    print("Step 3: Cleaning and preprocessing...")
    cleaned_descriptions = [clean_text(desc) for desc in descriptions]

    # Step 4: Extract features from the descriptions
    print("Step 4: Extracting features...")
    features = extract_features(cleaned_descriptions)

    # Step 5: Calculate similarity between items
    print("Step 5: Calculating similarity...")
    similarity_matrix = calculate_similarity(features)

    # Step 6: Rank similar items
    item_index = 0  # Index of the item for which you want to find similar items
    print("Step 6: Ranking similar items...")
    ranked_items = rank_similar_items(similarity_matrix, item_index)

    # Return ranked items as JSON response
    response = {
        "ranked_items": ranked_items
    }

    return json.dumps(response)
