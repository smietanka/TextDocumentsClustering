all_documents = [x.articleName + ' ' + x.articleContent for x in original_documents]
tfidf = TfidfVectorizer(tokenizer=PrepareDocument)
tfidfMatrix = tfidf.fit_transform(all_documents)

def PrepareDocument(document):
    document = CleanDocument(document)
    tokenizedDoc = TokenizeDocument(document)
    removedDoc = RemoveStopWords(tokenizedDoc)
    stemmedDoc = StemDocument(removedDoc)
    return stemmedDoc

def TokenizeDocument(document):
    tokens = [word for sent in nltk.sent_tokenize(document) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if len(token) >= 5:
                filtered_tokens.append(token)
    return filtered_tokens

def StemDocument(tokenizedDoc):
    return [stemmer.stem(t) for t in tokenizedDoc if t]

def CleanDocument(document):
    document = document.replace("\\n", " ")
    document = re.sub("[^a-zA-Z ]", " " , document)
    return document.lower()

def RemoveStopWords(tokenizedDocument):
    removedStopWords = [word for word in tokenizedDocument if not word in stopwords]
    return removedStopWords