import spacy
from newsapi import NewsApiClient
import json
from datetime import datetime, timedelta

# Initialize SpaCy
nlp = spacy.load("en_core_web_sm")

# Initialize NewsAPI client
# Replace 'YOUR_API_KEY' with your actual NewsAPI key
newsapi = NewsApiClient(api_key='YOUR_API_KEY')

def fetch_articles(query, days=7, language='en', page_size=10):
    """Fetch articles from NewsAPI."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    articles = newsapi.get_everything(q=query,
                                      language=language,
                                      from_param=start_date.strftime('%Y-%m-%d'),
                                      to=end_date.strftime('%Y-%m-%d'),
                                      sort_by='relevancy',
                                      page_size=page_size)
    return articles['articles']

def extract_facts_from_text(text):
    """Extract facts from text using SpaCy."""
    doc = nlp(text)
    facts = []
    for sent in doc.sents:
        entities = [(ent.text, ent.label_) for ent in sent.ents]
        if len(entities) > 1:
            facts.append((sent.text.strip(), entities))
    return facts

def process_articles(articles):
    """Process articles and extract facts."""
    all_facts = []
    for article in articles:
        title = article['title']
        content = article['description'] or ''  # Use description if content is not available
        full_text = f"{title}. {content}"
        
        facts = extract_facts_from_text(full_text)
        if facts:
            all_facts.append({
                'source': article['source']['name'],
                'title': title,
                'url': article['url'],
                'facts': facts
            })
    return all_facts

def save_facts_to_json(facts, filename):
    """Save extracted facts to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(facts, f, ensure_ascii=False, indent=4)

def main():
    # Fetch articles
    query = "technology"  # You can change this to any topic
    articles = fetch_articles(query)
    
    # Process articles and extract facts
    all_facts = process_articles(articles)
    
    # Save facts to JSON file
    save_facts_to_json(all_facts, 'extracted_facts.json')
    
    print(f"Extracted facts from {len(all_facts)} articles and saved to 'extracted_facts.json'")

if __name__ == "__main__":
    main()
