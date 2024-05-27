mkdir .vector_cache && cd .vector_cache
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip wiki-news-300d-1M.vec.zip
rm wiki-news-300d-1M.vec.zip

wget https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip glove.6B.100d.txt glove.6B.200d.txt glove.6B.300d.txt

python -c "import gensim.downloader as api;api.load('word2vec-google-news-300',return_path=True)"
ln -s $HOME/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz .