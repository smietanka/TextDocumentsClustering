import csv

class RowFromFile(object):
    def __init__(self, cluster=None, articleName=None, articleContent=None):
        self.cluster= cluster
        self.articleName = articleName
        self.articleContent = articleContent
        
original_documents = []
with open("sciezka.csv", encoding="utf8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        if len(row) == 3:
            original_documents.append(RowFromFile(int(row[0]), row[1], row[2]))