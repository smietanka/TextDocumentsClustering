from nltk.corpus import stopwords
words = [word for word in words if word not in stopwords.words('english')]
print(words[:100])
['Python', 'interpreted', 'highlevel', 'programming', 'language', 'generalpurpose', 'programming', 'Created', 'Guido', 'van', 'Rossum', 'first', 'released', 'Python', 'design', 'philosophy', 'emphasizes', 'code', 'readability', 'syntax', 'allows', 'programmers', 'express', 'concepts', 'fewer', 'lines', 'code', 'notably', 'using', 'significant', 'whitespace']