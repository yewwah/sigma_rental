from sklearn.base import BaseEstimator, TransformerMixin
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk


class TextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, max_features):
        self.tfidfVectorizer = TfidfVectorizer(use_idf=False, stop_words='english',
                                               tokenizer=self._custom_tokenizer, analyzer='word',
                                               max_features=max_features)
        self._vectorizer = None
        self._column = column

    def _custom_tokenizer(self, string):
        # string = re.sub('^[\w]', '', string)
        tokens = nltk.word_tokenize(string)
        cleaned = [x if not x.isdigit() else '_NUM_' for x in tokens]
        return [str(x.encode('utf-8')) for x in cleaned if (x.isalpha() or x == '_NUM_')]

    def _clean_html_tags(self, content):
        return BeautifulSoup(content, 'lxml').text

    def fit(self, df, y = None):
        if self._column == 'features':
            df[self._column] = df[self._column].apply(lambda x : ' '.join(x))
        self._vectorizer = self.tfidfVectorizer.fit(df[self._column].apply(self._clean_html_tags))
        return self
    
    def transform(self, df, y = None):
        return self._vectorizer.transform(df[self._column])

class ColumnExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, cols):
        self.cols = cols
    
    def transform(self, df, y = None):
        return df[self.cols].values
    
    def fit(self, X, y=None):
        return self
