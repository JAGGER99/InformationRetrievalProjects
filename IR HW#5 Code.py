
import math
import json
import string
from matplotlib import pyplot as plt

#Problem 3:

#part a:

def JM_score(Lambda, background_ULM_dict, query, document):

    
    # convert strings to Sets:
    query_set = set(query.split())
    document_set = set(document.split())

    # find list of words that are in BOTH Query and Document:
    common_words = list(query_set.intersection(document_set))
    #print(common_words)

    common_words_query_count_dict = {}  # key = word ... value = count in query
    common_words_document_count_dict = {} #key = word ... value = count in document


    # find c(w,Q) for each word found in BOTH Query and Document
    for word in common_words:
        common_words_query_count_dict[word] = common_words_query_count_dict.setdefault(word,0) + query.split().count(word)

    # find c(w,D) for each word found in BOTH Query and Document
    for word in common_words:
        common_words_document_count_dict[word] = common_words_document_count_dict.setdefault(word,0) + document.split().count(word)


    # find size of Document (number of words)
    document_size = len(document.split())
    
    #calculating the score for the given query + document pair:
    jm_score = 0

    for word in common_words:
        jm_score += common_words_query_count_dict[word] * math.log2(1 + (((1-Lambda) * common_words_document_count_dict[word]) / (Lambda * background_ULM_dict[word] * document_size)))

    return jm_score


#############################################################################################################################################################################

def Dirichlet_score(Mu, background_ULM_dict, query, document):


    # convert strings to Sets:
    query_set = set(query.split())
    document_set = set(document.split())

    # find list of words that are in BOTH Query and Document:
    common_words = list(query_set.intersection(document_set))
    print(common_words)

    
    common_words_query_count_dict = {}  # key = word ... value = count in query
    common_words_document_count_dict = {} #key = word ... value = count in document

    # find c(w,Q) for each word found in BOTH Query and Document
    for word in common_words:
        common_words_query_count_dict[word] = common_words_query_count_dict.setdefault(word,0) + query.split().count(word)

    # find c(w,D) for each word found in BOTH Query and Document
    for word in common_words:
        common_words_document_count_dict[word] = common_words_document_count_dict.setdefault(word,0) + document.split().count(word)


    # find size of Document (number of words)
    document_size = len(document.split())

    # find the size of Query (number of words)
    query_size = len(query.split())
    
    #calculating the score for the given query + document pair:
    dirichlet_score = 0

    for word in common_words:
        dirichlet_score += common_words_query_count_dict[word] * math.log2(1 + (common_words_document_count_dict[word] / (Mu * background_ULM_dict[word]))) - (query_size * math.log2(document_size + Mu))

    return dirichlet_score


###########################################################################################################################################################################################################
def generate_avg_score(smoothing_parameter, background_ULM_dict, queries, documents, scoring_method):
    
    query_avg_scores = [] # the average score of the top 5 docuements for the ith query

    # getting the jm score for each document query pair and storing in query_JM_scores:
    for query_dict in queries:
        
        scores_list = []
        sorted_doc_ids = []

        for doc_dict in documents:
            query = str(query_dict['query']).translate(str.maketrans('', '', string.punctuation))
            document = str(doc_dict['body']).translate(str.maketrans('', '', string.punctuation))
            score = scoring_method(smoothing_parameter, background_ULM_dict, query, document)
            
            scores_list.append([doc_dict['id'], score])

        # sorting the docs on the jm score:
        top_5_sorted_scores_list = sorted(scores_list,key=lambda l:l[1], reverse=True)[:5]

        # just want the doc id ...dont need the score any more:
        for pair in top_5_sorted_scores_list:
            sorted_doc_ids.append(pair[0])


        print(sorted_doc_ids)

        pos_sum = 0 # the cumulative sum of relevance scores for the current query
        avg_score = 0 # the average relevance score among the top 5 documents for the current query
        exists = False # boolean for keeping track if there exists a relavance value for a query/document pair

        for value in sorted_doc_ids:
            for rel_dict in relativity_scores:
                if(query_dict['query number'] == int(rel_dict['query_num']) and int(rel_dict['id']) == value):
                    pos_sum += rel_dict['position']
                    exists = True

            if(exists == False):
                pos_sum += 5
            exists = False

        avg_score = pos_sum / 5
        query_avg_scores.append(avg_score)
    
    return query_avg_scores
    


###################################################################################################################################################################################################
###################################################################################################################################################################################################

# main 

with open(r'C:\Users\JoshG\Documents\Information Retrieval HW Code\cranfield_data.json', 'r') as f1:
    # returns JSON object as a dictionary
    documents = json.loads(f1.read())

with open(r'C:\Users\JoshG\Documents\Information Retrieval HW Code\cran.qry.json', 'r') as f2:
    # returns JSON object as a dictionary
    queries = json.loads(f2.read())

with open(r'C:\Users\JoshG\Documents\Information Retrieval HW Code\cranqrel.json', 'r') as f3:
    # returns JSON object as a dictionary
    relativity_scores = json.loads(f3.read())


    # building the Reference language model for each word based off the entire collection of documents:
    
    data_dict = {}  # key: word ... value: count of word over total corpus

    total_words = 0 # number of total count of words in our corpus

    for dict in documents:
        for word in dict['body'].split():
            word = word.translate(str.maketrans('', '', string.punctuation))
            total_words += 1
            data_dict[word] = data_dict.get(word, 0) + 1

    background_ULM_dict = {} # key: word ... value: unigram language model value for 'word'
    
    # generating the background language model for each word in the corpus:
    for word in data_dict:
        background_ULM_dict[word] = data_dict.get(word, 0) + (data_dict[word] / total_words)

##########################################################################################################

    Lambda_list = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for i in range(len(Lambda_list)):
        query_avg_scores = generate_avg_score(Lambda_list[i], background_ULM_dict, queries, documents, JM_score)
        
        # plotting:
        plt.hist(query_avg_scores,  bins=[0, 1, 2, 3, 4, 5])
        plt.show()
    

    Mu_list = [100,500,1000,2000,4000,8000,10000]
    for smooth_parameter in Mu_list:
        query_avg_scores = generate_avg_score(smooth_parameter, background_ULM_dict, queries, documents, Dirichlet_score)

        # plotting:
        plt.hist(query_avg_scores, bins=[0, 1, 2, 3, 4, 5])
        plt.show()

