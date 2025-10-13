from textblob import TextBlob

with open("6-NLP/pride-and-prejudice.txt", encoding="utf8") as f:
    file_contents = f.read()

positives = []
negatives = []

blob = TextBlob(file_contents)

for sentence in blob.sentences:
    if sentence.polarity == 1:
        positives.append(sentence)
    elif sentence.polarity == -1:
        negatives.append(sentence)

for negative in negatives:
    print(negative)

for positive in positives:
    print(positive)

print(len(positives))
print(len(negatives))
