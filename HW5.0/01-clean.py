import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras import preprocessing
from tensorflow.keras.utils import to_categorical
maxlen       = 250
novel = []
label = []
with open('./The Communist Manifesto.txt', 'r') as f:
    lines = f.readlines()
    tcm = [0,0,0]
    lines = [x.strip() for x in lines if x.strip()!='']
    for index, line in enumerate(lines, 1):
        #print(index, line)
        novel.append(line)
        label.append(0)


with open('./Andersen\'s Fairy Tales.txt', 'r') as f:
    lines = f.readlines()
    aft = [0,0,1]
    lines = [x.strip() for x in lines if x.strip()!='']
    for index, line in enumerate(lines, 1):
        #print(index, line)
        novel.append(line)
        label.append(1)

with open('./The Project Gutenberg EBook of The United States\' Constitution.txt', 'r') as f:
    lines = f.readlines()
    frank = [0,1,0]
    lines = [x.strip() for x in lines if x.strip()!='']
    for index, line in enumerate(lines, 1):
        #print(index, line)
        novel.append(line)
        label.append(2)

label = to_categorical(label)
corpus = novel

#-----------------------------
#BASIC TOKENIZATION EXAMPLE
#-----------------------------

A=corpus[0]


temp=[]
for char in A: temp.append(char)





#-----------------------------
#DOCUMENT TERM MATRIX
#-----------------------------
#FREQUENCY COUNT MATRIX

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus) #WILL EXCLUDE WORDS OF LENGTH=1
# print("VOCABULARY-1",vectorizer.get_feature_names())
# print("DOCUMENT TERM MATRIX")
# print(X.toarray())

#-----------------------------
#FORM DICTIONARY AND ENCODE AS INDEX TOKENS
#-----------------------------

def form_dictionary(samples):
    token_index = {};
    #FORM DICTIONARY WITH WORD INDICE MAPPINGS
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1

    transformed_text=[]
    for sample in samples:
        tmp=[]
        for word in sample.split():
            tmp.append(token_index[word])
        transformed_text.append(tmp)

    # print("CONVERTED TEXT:", transformed_text)
    # print("VOCABULARY-2 (SKLEARN): ",token_index)
    return [token_index,transformed_text]

[vocab,x]=form_dictionary(corpus)


#-----------------------------
#VECTORIZE
#-----------------------------

#CHOLLET; IMDB (CHAPTER-3: P69)
def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# print(x[0]);
# print(x); print(len(x[0]))
x = vectorize_sequences(x,dimension=20000)
# print(x); #print(x); print(x.shape)


# #CHOLLET:  LISTING 6.1 WORD-LEVEL ONE-HOT ENCODING (TOY EXAMPLE)
def one_hot_encode(samples):
    #ONE HOT ENCODE (CONVERT EACH SENTENCE INTO MATRIX)
    max_length = 10
    results = np.zeros(shape=(len(samples),max_length,max(vocab.values()) + 1))
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = vocab.get(word)
            results[i, j, index] = 1.
    # print("ONE HOT")
    # print(results)

one_hot_encode(corpus)


#KERAS ONEHOT ENCODING
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=10)
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
novel = preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen,truncating='post')
one_hot_results = tokenizer.texts_to_matrix(corpus, mode='binary')
word_index = tokenizer.word_index
# print("KERAS")
# print(sequences)
#print(one_hot_results)
#print('Found %s unique tokens.' % len(word_index))


# ONE-HOT HASHING TRICK,
dimensionality = 10
max_length = 10
results = np.zeros((len(corpus), max_length, dimensionality))
for i, sample in enumerate(corpus):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.

# print("HASHING")
#print(results)

#save the data
np.savez('novel_data.npz',novel = novel,label = label)