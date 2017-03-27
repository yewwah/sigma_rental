from sklearn.base import BaseEstimator, TransformerMixin
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk


class TextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        max_features = 50
        self.tfidfVectorizer = TfidfVectorizer(use_idf=False, stop_words='english',
                                               tokenizer=self._custom_tokenizer, analyzer='word',
                                               max_features=max_features)
        self._vectorizer = None
        self._column = 'description'

    def _custom_tokenizer(self, string):
        # string = re.sub('^[\w]', '', string)
        tokens = nltk.word_tokenize(string)
        cleaned = [x if not x.isdigit() else '_NUM_' for x in tokens]
        return [str(x.encode('utf-8')) for x in cleaned if (x.isalpha() or x == '_NUM_')]

    def _clean_html_tags(self, content):
        return BeautifulSoup(content, 'lxml').text

    def fit(self, df):
        self._vectorizer = self.tfidfVectorizer.fit(df[self._column].apply(self._clean_html_tags))
        return self

    def transform(self, df):
        return self._vectorizer.transform(df[self._column]).todense()
