import os,re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,CountVectorizer

#nltk.download('stopwords')
#nltk.download('punkt')

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def tokenize_text(text):
    return word_tokenize(text)


def lemmatize_stem_text(tokens, lemmatize=True, stem=False):
    lemmatizer = WordNetLemmatizer() if lemmatize else None
    stemmer = PorterStemmer() if stem else None

    processed_tokens = []
    for token in tokens:
        if lemmatizer:
            token = lemmatizer.lemmatize(token)
        if stemmer:
            token = stemmer.stem(token)
        processed_tokens.append(token)

    return processed_tokens

def remove_stopwords(tokens, lang='english'):
    stop_words = set(stopwords.words(lang))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

def preprocess_pipeline(text, lang='english'):
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    filtered_tokens = remove_stopwords(tokens, lang)
    lemmatized_stemmed_tokens = lemmatize_stem_text(filtered_tokens)

    return ' '.join(lemmatized_stemmed_tokens)

def preprocess(config, train_val_file, tese_file, data_col):
    train_val_data = pd.read_csv(os.path.join(config.dataset,train_val_file))
    test_data = pd.read_csv(os.path.join(config.dataset,tese_file))
    
    train_val_data[data_col].apply(preprocess_pipeline)
    test_data[data_col].apply(preprocess_pipeline)
    texts = pd.concat([train_val_data,test_data])[data_col].tolist() # sentences after lemmatize & stem
    return texts, train_val_data, test_data
def BOW(config, train_val_file='train.csv', tese_file='test.csv',data_col="Sentence"):
    texts, train_val_data, test_data = preprocess(config, train_val_file, tese_file, data_col)
    
    vectorizer = CountVectorizer()
    vectorizer.fit(texts)
    config.input_dim = len(vectorizer.vocabulary_)
    return vectorizer, train_val_data, test_data

def TFIDF(config, train_val_file='train.csv', tese_file='test.csv',data_col="Sentence"):
    texts, train_val_data, test_data = preprocess(config, train_val_file, tese_file, data_col)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    config.input_dim = tfidf_matrix.shape[1]
    return vectorizer, train_val_data, test_data

def NGram(config, 
          train_val_file='train.csv', tese_file='test.csv',data_col="Sentence",
          min_n=1, max_n=2):
    texts, train_val_data, test_data = preprocess(config, train_val_file, tese_file, data_col)

    vectorizer = CountVectorizer(ngram_range=(min_n, max_n))
    X = vectorizer.fit_transform(texts)
    config.input_dim = X.shape[1]
    return vectorizer, train_val_data, test_data

if __name__ == '__main__':
    corpus = [
    "play something from 1971 by john bonham.",
    "i d like to give a two rating to the abolition of britain.",
    "where can i locate the show the return of mr moto.",
    "is there a game called the neutral zone."
    ]

    texts = [preprocess_pipeline(i) for i in corpus]
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    vector = vectorizer.transform(texts[:1])
    print(vector.toarray()[0])
    