text = 'Python is an interpreted high-level programming language for\n general-purpose programming. Created by Guido van Rossum and\n first released in 1991, Python has a design philosophy that emphasizes code readability, and a syntax that allows programmers to express\n concepts in fewer lines of code,[25][26] notably using significant whitespace.'
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
print(tokens[:100])

['Python', 'is', 'an', 'interpreted', 'high-level', 'programming', 'language', 'for', 'general-purpose', 'programming', '.', 'Created', 'by', 'Guido', 'van', 'Rossum', 'and', 'first', 'released', 'in', '1991', ',', 'Python', 'has', 'a', 'design', 'philosophy', 'that', 'emphasizes', 'code', 'readability', ',', 'and', 'a', 'syntax', 'that', 'allows', 'programmers', 'to', 'express', 'concepts', 'in', 'fewer', 'lines', 'of', 'code', ',', '[', '25', ']', '[', '26', ']', 'notably', 'using', 'significant', 'whitespace', '.']