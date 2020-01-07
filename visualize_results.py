import os 
import csv

# sentence
with open('./sentence_similarity.csv', newline='') as file:
    contents = list(csv.reader(file, delimiter=';'))

n = 66

lines = [[] for i in range(n)]
for i, info in enumerate(contents):
    if i < n:
        lines[i] = info
    else:
        lines[i % n].append(info[-1])

with open('./sentence_similarity_table.csv', mode='w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['Sentence_1', 'Sentence_2', 'Similarity by human judging', 'Datamuse', 'Word2Vec', 'Glove', 'FastText', 'Yago'])
    for line in lines:
        writer.writerow(line)

with open('./sentence_similarity.csv', newline='') as file:
    contents = list(csv.reader(file, delimiter=';'))

# word
