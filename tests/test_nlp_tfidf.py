from sklearn.feature_extraction.text import TfidfVectorizer

def test_tfidf():
    vec = TfidfVectorizer()
    X = vec.fit_transform(["app crash", "driver rude", "price too high"])
    assert X.shape[0] == 3
