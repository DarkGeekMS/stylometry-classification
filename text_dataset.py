import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

class TextDataset:
    def __init__(self, txt_list, norm=None, vectorizer='embed', w2v_path=None):
        self.txt_list = txt_list
        self.norm = norm
        self.vectorizer = vectorizer
        self.w2v_path = w2v_path
        self.embed_size = None
        self.max_len = 1
        self.bos = '<s>'
        self.eos = '</s>'

    def read_corpus(self, txt_list):
        sents = dict()
        for txt_file in txt_list:
            sents[txt_file[-13:-10]] = list()
            with open(txt_file) as f:
                for line in f:
                    line = line.strip()
                    sents[txt_file[-13:-10]].append(line)
        return sents

    def process_text(self, text_corpus):
        processed_corpus = dict()
        if self.norm:
            text_normalizer = PorterStemmer() if self.norm == 'stem' else WordNetLemmatizer()
        for key in text_corpus.keys():
            processed_list = list()
            for sent in text_corpus[key]:
                tokens = word_tokenize(sent)
                if self.norm == 'stem':
                    tokens = [text_normalizer.stem(word) for word in tokens]
                elif self.norm == 'lemma':
                    tokens = [text_normalizer.lemmatize(word) for word in tokens]
                processed_list.append(tokens)
            processed_corpus[key] = processed_list
        return processed_corpus

    def list_data(self, text_corpus):
        label_encoder = preprocessing.LabelEncoder()
        classes = list(text_corpus.keys())
        labels = label_encoder.fit_transform(classes)
        X, Y = list(), list()
        for key, label in zip(classes, labels):
            for sent in text_corpus[key]:
                X.append(sent)
                Y.append(label)
        return X, Y

    def get_word_dict(self, text_list):
        word_dict = {}
        for sent in text_list:
            if len(sent) > self.max_len-2:
                self.max_len = len(sent) + 2
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                if not self.embed_size:
                    self.embed_size = np.fromstring(vec, sep=' ').shape[0]
        print('Found (%s/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def vectorize_text(self, xtrain, xvalid):
        vectorized_xtrain = list()
        vectorized_xvalid = list()
        if self.vectorizer == 'count':
            tfidf_vec = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', 
                                    analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3),
                                    use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words = 'english')
            tfidf_vec.fit(list(xtrain) + list(xvalid))
            vectorized_xtrain = tfidf_vec.transform(xtrain)
            vectorized_xvalid = tfidf_vec.transform(xvalid)
        elif self.vectorizer == 'tfidf':
            count_vec = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                                    ngram_range=(1, 3), stop_words = 'english')
            count_vec.fit(list(xtrain) + list(xvalid))
            vectorized_xtrain = count_vec.transform(xtrain)
            vectorized_xvalid = count_vec.transform(xvalid)
        elif self.vectorizer == 'embed':
            word_dict = self.get_word_dict(list(xtrain) + list(xvalid))
            word_vec = self.get_w2v(word_dict)
            for sent in xtrain:
                sent_embed = np.zeros((self.max_len, self.embed_size))
                for i in range(len(sent)):
                    if sent[i] in word_vec:
                        sent_embed[i, :] = word_vec[sent[i]]
                vectorized_xtrain.append(sent_embed)
            for sent in xvalid:
                sent_embed = np.zeros((self.max_len, self.embed_size))
                for i in range(len(sent)):
                    if sent[i] in word_vec:
                        sent_embed[i, :] = word_vec[sent[i]]
                vectorized_xvalid.append(sent_embed)
        return vectorized_xtrain, vectorized_xvalid

    def build_dataset(self, eval=False):
        text_corpus = self.read_corpus(self.txt_list)
        if self.vectorizer == 'embed':
            text_corpus = self.process_text(text_corpus)
        X, Y = self.list_data(text_corpus)
        if not eval:
            xtrain, xvalid, ytrain, yvalid = train_test_split(X, Y, stratify=Y, random_state=42,
                                                            test_size=0.1, shuffle=True)
            xtrain, xvalid = self.vectorize_text(xtrain, xvalid)
            return xtrain, xvalid, ytrain, yvalid
        else:
            X, _ = self.vectorize_text(X, list())
            return X, None, Y, None
