table = text.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]
print(stripped[:100])
['Python', 'is', 'an', 'interpreted', 'highlevel', 'programming', 'language', 'for', 'generalpurpose', 'programming', '', 'Created', 'by', 'Guido', 'van', 'Rossum', 'and', 'first', 'released', 'in', '1991', '', 'Python', 'has', 'a', 'design', 'philosophy', 'that', 'emphasizes', 'code', 'readability', '', 'and', 'a', 'syntax', 'that', 'allows', 'programmers', 'to', 'express', 'concepts', 'in', 'fewer', 'lines', 'of', 'code', '', '', '25', '', '', '26', '', 'notably', 'using', 'significant', 'whitespace', '']
words = [word for word in stripped if word.isalpha()]
print(words[:100])
['Python', 'is', 'an', 'interpreted', 'highlevel', 'programming', 'language', 'for', 'generalpurpose', 'programming', 'Created', 'by', 'Guido', 'van', 'Rossum', 'and', 'first', 'released', 'in', 'Python', 'has', 'a', 'design', 'philosophy', 'that', 'emphasizes', 'code', 'readability', 'and', 'a', 'syntax', 'that', 'allows', 'programmers', 'to', 'express', 'concepts', 'in', 'fewer', 'lines', 'of', 'code', 'notably', 'using', 'significant', 'whitespace']