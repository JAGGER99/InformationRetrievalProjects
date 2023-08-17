

import random

random.seed(10)

def EM_step(thetas, pis):


    hidden = list()  
    for i in range(N):
        hidden.append(dict())
        doc = CORPUS[i]
        for word in doc:
            hidden[-1][word] = list()
            denom = LAMBDA * BKG[word]

            for k in range(K):

                q = (1-LAMBDA) * (thetas[word][k]) * (pis[k][i])
                hidden[-1][word].append(q)
                denom += q

            for k in range(K):
                hidden[-1][word][k] /= denom      # [doc_number] # [word] # [topic_number]



    # generate ndks[document][topic]
    ndks = list()
    for i in range(N):
        ndks.append(list())
        doc = CORPUS[i]
        for k in range(K):
            n_dk = 0
            for word in doc:
                n_dk += doc[word] * hidden[i][word][k]
            ndks[-1].append(n_dk)


    # generate nwks[word][topic]
    nwks = {word:[0 for i in range(K)] for word in BKG}
    for i in range(N):
        doc = CORPUS[i]
        for k in range(K):
            for word in doc:
                nwks[word][k] += doc[word] * hidden[i][word][k]


    return ndks,nwks




################################################################################################

with open(r'Documents\Information Retrieval HW Code\dblp-small.txt') as docs:
    corpus = docs.read()
    K = 20 # number of topics
    LAMBDA = 0.9  # split parameter
    list_of_docs = corpus.split('\n')  # list of documents
    N = len(list_of_docs) # # number of documents = 10836
    
    CORPUS = []  # list of dictionaries

    for i in range(N):

        CORPUS.append(dict())  #key: word    #value: count(word in this current doc)/words in doc  
        count_words = 0
        
        for word in list_of_docs[i].split():
            count_words += 1
            CORPUS[i][word] = CORPUS[i].get(word, 0) + 1

        for word in CORPUS[i]:
            CORPUS[i][word] /= count_words   



    total_text = corpus.replace('\n', ' ')
    total_words = total_text.split()
    unique_words = set(total_words)
    
    

    #generating the background LM for each word:
    ############################################################################################
    word_counts = {}  # key: word ... value: count of word over total corpus

    for word in total_words:      
        word_counts[word] = word_counts.get(word, 0) + 1
        

    BKG = {} # key: word ... value: unigram language model value for 'word'
    
    # generating the background language model for each word in the corpus:
    for word in word_counts:
        BKG[word] = word_counts.get(word, 0) + (word_counts[word] / len(total_words)) 

    ############################################################################################
    
    thetas = {word:[random.random() for i in range(K)] for word in BKG}      
    pis = [[random.random() for n in range(N)] for j in range(K)]

    old_log_likelihood = 0

    #Run 20 times
    for i in range(20):

        ndks, nwks = EM_step(thetas, pis)

        # setting old pi and theta:
        old_pis = thetas
        old_thetas = pis

        #pis
        for i in range(N):
            denom = sum(ndks[i]) + 0.001
            for k in range(K):
                pis[k][i] = ndks[i][k]/denom

        #thetas
        for k in range(K):
            denom = 0.001
            for word in BKG:
                denom += nwks[word][k]

            for word in BKG:
                thetas[word][k] = nwks[word][k] / denom

    
""" 
Comments:

    With lambda set to 0.9 we have more emphasis placed on the topics rather than the background model! 
    The larger lambda was the faster the model converged since we focused more on the constant background model!
    If someone wanted to find descriptive top ten words, then they should have a higher lambda in order to focus on the topic model over the background model.
    
"""