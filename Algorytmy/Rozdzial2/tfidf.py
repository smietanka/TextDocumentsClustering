import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
def display_scores(vectorizer, tfidf_result):
    scores = zip(vectorizer.get_feature_names(), np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    allTerms = ["{0} ({1})".format(z[0], z[1]) for z in sorted_scores]
    for term in allTerms:
        print(term)
text1 = "The game of life is a game of everlasting learning"
text2 = "The unexamined life is not worth living"
text3 = "Never stop learning"
tfidf = TfidfVectorizer()
tfidfMatrix = tfidf.fit_transform([text1, text2, text3])
display_scores(tfidf, tfidfMatrix)

learning (0.699736372829852)
never (0.6227660078332259)
stop (0.6227660078332259)
game (0.5946064659329678)
of (0.5946064659329678)
is (0.5436769522249277)
life (0.5436769522249277)
the (0.5436769522249277)
living (0.4175666238781924)
not (0.4175666238781924)
unexamined (0.4175666238781924)
worth (0.4175666238781924)
everlasting (0.2973032329664839)