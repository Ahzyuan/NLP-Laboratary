import os,re,torch,sys
import pandas as pd
import numpy as np
from torchtext.vocab import GloVe
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,CountVectorizer

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def tokenize_text(text):
    return word_tokenize(text)


def lemmatize_stem_text(tokens, lemmatize=True, stem=True):
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

def GLOVE(config, 
          train_val_file='train.csv', tese_file='test.csv',
          data_col="Sentence"):
    texts, train_val_data, test_data = preprocess(config, train_val_file, tese_file, data_col)
    
    print('Loading word embedding model for Glove...',end='')
    glove = GloVe(name='6B', dim=50)
    glove_vocab = {word.lower(): vec for word, vec in zip(glove.stoi.keys(), glove.vectors)}
    print('Done!')

    config.input_dim = glove_vocab.vectors.shape[1]

    def sent2vec(text, embeddings):
        words = text.split() 
        vectors = [glove_vocab.get(word.lower(),torch.zeros(embeddings.vectors.shape[1])) for word in words] 
        res = torch.stack(vectors, dim=0)
        res = torch.mean(res,dim=0)
        return res
        
    return lambda x: sent2vec(x, glove_vocab), train_val_data, test_data

def WORD2VEC(config, 
             train_val_file='train.csv', tese_file='test.csv',
             data_col="Sentence",
             word2vec_model=os.path.join(sys.path[0],'.vector_cache','word2vec-google-news-300.gz')):
    texts, train_val_data, test_data = preprocess(config, train_val_file, tese_file, data_col)
    print('Loading word embedding model for Word2Vec...',end='')
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model, binary=True)
    print('Done!')
    
    config.input_dim = word2vec_model.vector_size

    def sent2vec(text, embeddings):
        words = text.split() 
        vectors = [torch.tensor(embeddings[word.lower()]) if word.lower() in embeddings else torch.zeros(embeddings.vectors.shape[1]) for word in words] 
        res = torch.stack(vectors, dim=0)
        res = torch.mean(res,dim=0)
        return res
        
    return lambda x: sent2vec(x, word2vec_model), train_val_data, test_data

def FASTTEXT(config, 
             train_val_file='train.csv', tese_file='test.csv',
             data_col="Sentence",
             fasttext_model=os.path.join(sys.path[0],'.vector_cache','wiki-news-300d-1M.vec')):
    texts, train_val_data, test_data = preprocess(config, train_val_file, tese_file, data_col)
    print('Loading word embedding model for FastText...',end='')
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model, binary=False)
    print('Done!')
    
    config.input_dim = word2vec_model.vector_size

    def sent2vec(text, embeddings):
        words = text.split() 
        vectors = [torch.tensor(embeddings[word.lower()]) if word.lower() in embeddings else torch.zeros(embeddings.vectors.shape[1]) for word in words] 
        res = torch.stack(vectors, dim=0)
        res = torch.mean(res,dim=0)
        return res
        
    return lambda x: sent2vec(x, word2vec_model), train_val_data, test_data

if __name__ == '__main__':
    corpus = [
    "play something from 1971 by john bonham.",
    "i d like to give a two rating to the abolition of britain.",
    "where can i locate the show the return of mr moto.",
    "is there a game called the neutral zone."
    ]

    texts = [preprocess_pipeline(i) for i in corpus]
    print(texts)
    word2vec_model=os.path.join(sys.path[0],'.vector_cache','wiki-news-300d-1M.vec')
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model, binary=False)
    def sent2vec(text, embeddings):
        words = text.split() 
        vectors = [torch.tensor(embeddings[word.lower()]) if word.lower() in embeddings else torch.zeros(embeddings.vectors.shape[1]) for word in words] 
        res = torch.stack(vectors, dim=0)
        res = torch.mean(res,dim=0)
        return res

    vectorizer = lambda x: sent2vec(x, word2vec_model)
    vector = vectorizer(texts[0])
    print(vector)
    