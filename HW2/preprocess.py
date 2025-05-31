import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
import re

def remove_stopwords(text: str) -> str:
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text

def remove_HTML(text: str) -> str:
    rmHTML = re.sub("<[^>]+>", "", text).strip()

    return rmHTML

def lemmatization(text: str) -> str:
    filter = nltk.stem.wordnet.WordNetLemmatizer()
    text = text.split(" ")
    lemed = []
    for i in text:
        lem = filter.lemmatize(i, "n")
        if(lem == i): lem = filter.lemmatize(i, "v")
        lemed.append(lem)
    
    preprocessed_text = ' '.join(lemed)

    return preprocessed_text
    

def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)
    preprocessed_text = remove_HTML(preprocessed_text)
    preprocessed_text = lemmatization(preprocessed_text)

    return preprocessed_text