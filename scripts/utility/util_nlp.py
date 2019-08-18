#======================================================
# NLP Utility Functions
#======================================================
'''
Info:       Utility functions for NLP.
Version:    2.0
Author:     Young Lee
Created:    Saturday, 13 April 2019
'''
# Import modules
import re
import string
from collections import defaultdict

import nltk
import numpy as np
import pandas as pd
import sparse_dot_topn.sparse_dot_topn as ct
from gensim.parsing.preprocessing import STOPWORDS
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.stem.porter import *
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')


#------------------------------
# Utility Functions
#------------------------------
def preprocess_text(x):
    no_special_chars = re.sub('\W+',' ', x.lower().strip())
    return ' '.join(re.split('(\d+)', no_special_chars))

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y=None):
        tfidf = TfidfVectorizer(stop_words = 'english')
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        trans_array = np.array([
            np.mean([np.array(self.word2vec[w]).astype(np.float) * self.word2weight[w].astype(np.float) 
                        for w in words.split() if w in self.word2vec and len(w) > 2] or
                    [np.zeros(self.dim)], axis=0)
            for words in X.values
        ])

        return trans_array


def generate_w2v(glove_model_path):
    with open(glove_model_path, encoding='utf-8') as glove:
        array_of_rows_ = glove.readlines()
        words = pd.DataFrame([row.split(" ") for row in array_of_rows_[0:]])
        
        try:
            words = words.apply(pd.to_numeric, errors='ignore')
        except Exception as e:
            print("Words apply  to numeric failed. Details: " + str(e))

        try:
            d = {'word': words.loc[:,0].tolist(), 'embedding': words.loc[:,1:].values.tolist()}
        except Exception as e:
            print("Creation of d failed. Details: " + str(e))

        try:
            d50 = pd.DataFrame(data=d)
        except Exception as e:
            print("Creation of d50 df failed. Details: " + str(e))

        try:
            w2v = d50.set_index('word')['embedding'].to_dict()
        except Exception as e:
            print("Creation of w2v failed. Details: " + str(e))

        return w2v


#------------------------------
# TF-IDF and N-Grams
#------------------------------
# Clean string
def clean_string(string_, verbose=1):
    # Remove tickers
    no_tickers=re.sub(r'\$\w*','',string_)
    temp_word_list = word_tokenize(no_tickers)
    # Remove stopwords
    #list_no_stopwords=[i for i in temp_word_list if i.lower() not in cache_english_stopwords]
    # Remove hyperlinks
    #list_no_hyperlinks=[re.sub(r'https?:\/\/.*\/\w*','',i) for i in list_no_stopwords]
    list_no_hyperlinks=[re.sub(r'https?:\/\/.*\/\w*','',i) for i in temp_word_list]
    # Remove hashtags
    list_no_hashtags=[re.sub(r'#', '', i) for i in list_no_hyperlinks]
    # Remove Punctuation and split 's, 't, 've with a space for filter
    list_no_punctuation=[re.sub(r'['+string.punctuation+']+', ' ', i) for i in list_no_hashtags]
    # Remove multiple whitespace
    new_sent = ' '.join(list_no_punctuation)
    filtered_list = word_tokenize(new_sent)
    filtered_sent =' '.join(filtered_list)
    clean_sent=re.sub(r'\s\s+', ' ', filtered_sent)
    # Remove any whitespace at the front of the sentence
    clean_sent=clean_sent.lstrip(' ')
    if verbose==1:
        print('No tickers:')
        print(no_tickers)
        print('Temp_list:')
        print(temp_word_list)
        # print('No Stopwords:')
        # print(list_no_stopwords)
        print('No hyperlinks:')
        print(list_no_hyperlinks)
        print('No hashtags:')
        print(list_no_hashtags)
        print('No punctuation:')
        print(list_no_punctuation)
        print('Clean list of words:')
        print(filtered_list)
        print('Clean sentence:')
        print(clean_sent.lower())
    return clean_sent.lower()

def preprocess_text(text):
    # Filter out useless symbols
    useless_symbols = ['.', ',', '_', "'"]
    for symbol in useless_symbols:
        text = text.replace(symbol, '')
    # Remove double == symbols
    text = re.sub(r"=+"," ",text)
    # Remove punctuation 's
    text = re.sub(r"'s|’s","",text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Filter out numbers and lowercase
    tokens = [word.lower() for word in tokens if not word.isnumeric()]
    # Stopword removal
    tokens = [word for word in tokens if word not in STOPWORDS]
    # Part of speech tag (pos tag)
    #nltk.pos_tag(tokens)
    # Lemmatize
    lemmy = WordNetLemmatizer()
    tokens = [lemmy.lemmatize(word, pos='v') for word in tokens]
    #' '.join([lemmy.lemmatize(word, pos='v') for word in tokens])
    # Stemming
    stemmer = SnowballStemmer(language='english')
    tokens = [stemmer.stem(word) for word in tokens]
    # Remove '-'
    tokens = [word for word in tokens if word != '–']
    tokens = [word for word in tokens if word != '-']
    # Replace digits followed by units with just units
    tokens = [word if len(re.findall(r'\d+\S+', word))==0 else re.sub(r'\d+', '',word) for word in tokens]
    #' '.join([stemmer.stem(word) for word in tokens])
    return tokens

def ngrams(string, n=4):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

# print('All 4-grams in "McDonalds":')
# ngrams('McDonalds', n=4)

def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
    idx_dtype = np.int32 
    nnz_max = M*ntop
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)
    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)
    return csr_matrix((data,indices,indptr),shape=(M,N))

# def get_matches_df(sparse_matrix, all_titles, top=100):
#     non_zeros = sparse_matrix.nonzero()
#     sparserows = non_zeros[0]
#     sparsecols = non_zeros[1]
#     if top:
#         nr_matches = min(top, sparsecols.size)
#         print('Taking at most', nr_matches, 'matches')
#     else:
#         nr_matches = sparsecols.size
#         print('Taking at most', nr_matches, 'matches')
#     left_side = np.empty([nr_matches], dtype=object)
#     right_side = np.empty([nr_matches], dtype=object)
#     similarity = np.zeros(nr_matches)
#     for index in range(0, nr_matches):
#         left_side[index] = all_titles.iloc[sparserows[index]]
#         right_side[index] = all_titles.iloc[sparsecols[index]]
#         similarity[index] = sparse_matrix.data[index]
#     return pd.DataFrame({'left_idx': sparserows[:nr_matches],
#                         'right_idx': sparsecols[:nr_matches],
#                         'left': left_side,
#                         'right': right_side,
#                         'similarity': similarity})

def get_matches_df(sparse_matrix, titles_a, titles_b, top=None):
    non_zeros = sparse_matrix.nonzero()
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    if top:
        nr_matches = min(top, sparsecols.size)
        print('Taking at most', nr_matches, 'matches')
    else:
        nr_matches = sparsecols.size
        print('Taking at most', nr_matches, 'matches')
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similarity = np.zeros(nr_matches)
    for index in range(0, nr_matches):
        left_side[index] = titles_a.iloc[sparserows[index]]
        right_side[index] = titles_b.iloc[sparsecols[index]]
        similarity[index] = sparse_matrix.data[index]
    return pd.DataFrame({'left_idx': sparserows[:nr_matches],
                        'right_idx': sparsecols[:nr_matches],
                        'left': left_side,
                        'right': right_side,
                        'similarity': similarity})


#------------------------------
# Wiki article cleansing
#------------------------------
# Wiki remove headers
def remove_wiki_headers(text, bad_header_list, verbose=True):
    headers = re.findall(r'\n== .*? ==\n',text.lower())
    header_index = [text.lower().index(header) for header in headers]
    try: 
        header_index[0] = 0
        bad_header_index = [header_index[i] for i, header in enumerate(headers) if re.sub(r'\n|= | =|=','',header.lower()) in bad_header_list]

        text_ = ''
        for start_idx, end_idx in zip(header_index, header_index[1:] + [None]):
            if start_idx not in bad_header_index:
                text_ = text_ + text[start_idx:end_idx]
            elif verbose:
                removed_header = re.sub(r'\n|= | =|=','',headers[header_index.index(start_idx)])
                print('Removed header:', removed_header)
        return text_
    except IndexError:
        return text

# Clean wiki article
def clean_wiki_text(text):
    # Remove text written between double curly brackets
    text = re.sub(r"{.*}","",text)
    # Remove file attachments
    text = re.sub(r"\[\[File:.*\]\]","",text)
    # Remove image attachments
    text = re.sub(r"\[\[Image:.*\]\]","",text)
    # Remove unwanted lines starting from special characters
    text = re.sub(r"\n: \'\'.*","",text)
    text = re.sub(r"\n!.*","",text)
    text = re.sub(r"^:\'\'.*","",text)
    # Remove non-breaking space symbols
    text = re.sub(r"&nbsp","",text)
    # Remove URLs link
    text = re.sub(r"http\S+","",text)
    # Remove digits from text
    #text = re.sub(r"\d+","",text)
    # Remove text written between small braces   
    text = re.sub(r"\(.*\)","",text)
    # Remove the sentences inside infobox or taxobox
    text = re.sub(r"\| .*","",text)
    text = re.sub(r"\n\|.*","",text)
    text = re.sub(r"\n \|.*","",text)
    text = re.sub(r".* \|\n","",text)
    text = re.sub(r".*\|\n","",text)
    # Remove text written between angle bracket
    text = re.sub(r"","",text)
    # Remove new line character
    text = re.sub(r"\n"," ",text)
    # Remove all punctuations
    text = re.sub(r"\"|\#|\$|\%|\'|\(|\)|\*|\/|\|\?|\?|\@|\[|\\|\]|\^|\`|\{|\||\}","",text)
    #text = re.sub(r"\!|\&|\+|\,|\-|\.|\:|\;|\_|\~"," ",text)
    text = re.sub(r"\!|\&|\+|\,|\.|\:|\;|\_|\~"," ",text)
    # Replace consecutive multiple space
    text = re.sub(r" +"," ",text)
    # Replace consecutive multiple dashes
    text = re.sub(r"-+","-",text)
    return text

# Remove
def filter_bad_words(text, bad_words=None):
    if bad_words:
        pass
    else:
        bad_words   = ['example', 'chapter', 'fig', 'aaaaaaaaaaaaaaaaaaaaa', 'introduction', 'intro', '×××××', '©']
    text        = ' '.join([word for word in text.split(' ') if not word.lower() in bad_words])
    return text

# Get sub category positions
def get_wiki_categories(text):
    sub_topics = re.findall(r' === .*? === ',text)
    sub_topics_idx = [text.index(sub_topic) for sub_topic in sub_topics]

    headers = re.findall(r' == .*? == ',text)
    headers_idx = [text.index(header) for header in headers]

    header_df = pd.DataFrame([headers, headers_idx, headers_idx[1:] + [len(text)]]).T
    header_df.columns = ['header', 'header_start_idx', 'header_end_idx']
    sub_topics_df = []
    for sub_topic, sub_topic_idx in zip(sub_topics, sub_topics_idx):
        for i, row in header_df.iterrows():
            if (sub_topic_idx > row['header_start_idx']) & (sub_topic_idx < row['header_end_idx']):
                sub_topics_df.append(row['header'])
    sub_topics_df = pd.DataFrame([sub_topics, sub_topics_df, sub_topics_idx]).T
    sub_topics_df.columns = ['sub_topic', 'header', 'sub_topic_start_idx']
    sub_topics_df = pd.merge(sub_topics_df, header_df, on='header', how='inner')
    sub_topics_df.loc[:, 'sub_topic_end_idx'] = sub_topics_df.loc[:, 'sub_topic_start_idx'].shift(-1)
    sub_topics_df.loc[:, 'sub_topic_end_idx'] = sub_topics_df[['sub_topic_end_idx', 'header_end_idx']].min(axis=1).astype(int)
    sub_topics_df = sub_topics_df[['sub_topic', 'header', 'sub_topic_start_idx', 'sub_topic_end_idx', 'header_start_idx',
        'header_end_idx']]
    return sub_topics_df
