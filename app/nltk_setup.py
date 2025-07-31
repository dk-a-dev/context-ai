import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
    print("NLTK 'punkt' resource already present.")
except LookupError:
    print("Downloading NLTK 'punkt' resource...")
    nltk.download('punkt_tab')
