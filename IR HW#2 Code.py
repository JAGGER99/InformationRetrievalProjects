
""" 
Author: Josh Greene
Date: 8/30/22
Class: Information Retrieval 
Professor: Santu 
Homework #2 d & e
"""


#imports:
from itertools import combinations
import math

# Information Gain for a pair given an occurance dictionary and number of docs:
def Information_Gain(indv_dict, cooccur_dict, numOfDocs, pair):
    
    word1 = pair[0] # X
    word2 = pair[1] # Y
    
    Prob_x =  (len(indv_dict[word1]) + 0.5) / (numOfDocs + 1)   # P(X = 1)
    Prob_y = (len(indv_dict[word2]) + 0.5) / (numOfDocs + 1)    # P(Y = 1)

    Prob_not_x =  1 - Prob_x   # P(X = 0)
    Prob_not_y = 1 - Prob_y    # P(Y = 0)

    Prob_x_and_not_y =  (len(indv_dict[word1]) - cooccur_dict[pair]) / (numOfDocs + 1)  # P(X = 1, Y = 0)
    Prob_not_x_and_y =  (len(indv_dict[word2]) - cooccur_dict[pair]) / (numOfDocs + 1)  # P(X = 0 , Y = 1)
    
    Prob_x_and_y = (cooccur_dict[pair] + 0.25) / (numOfDocs + 1)  # P(X = 1, Y = 1) 
    Prob_not_x_not_y = 1 - ((len(indv_dict[word1]) + len(indv_dict[word2]) - cooccur_dict[pair] + 0.75) / (numOfDocs + 1))  # P(X = 0, Y = 0)
    

    # I(0,0)
    if(Prob_not_x_not_y / (Prob_not_y * Prob_not_x) > 0):
        IG1 = Prob_not_x_not_y * math.log2(Prob_not_x_not_y / (Prob_not_y * Prob_not_x))
    else:
        IG1 = 0 # log(0) == True

    # I(0,1)
    if(Prob_not_x_and_y / (Prob_not_x * Prob_y) > 0):
        IG2 = Prob_not_x_and_y * math.log2(Prob_not_x_and_y / (Prob_not_x * Prob_y))
    else:
        IG2 = 0 # log(0) == True

    # I(1,0)
    if(Prob_x_and_not_y / (Prob_x * Prob_not_y) > 0):
        IG3 = Prob_x_and_not_y * math.log2(Prob_x_and_not_y / (Prob_x * Prob_not_y))
    else:
        IG3 = 0 # log(0) == True
        
    # I(1,1)
    if(Prob_x_and_y / (Prob_x * Prob_y) > 0): 
        IG4 = Prob_x_and_y * math.log2(Prob_x_and_y / (Prob_x * Prob_y))
    else:
        IG4 = 0  # log(0) == True

    # total IG for the current pair:
    IG_sum = IG1 + IG2 + IG3 + IG4
    
    return IG_sum

#####################################################################################################

numOfDocs = 0 # # the number of documents (the number of lines in the txt file)

indv_cnt_dict = {}  # dictionary for holding unique words and the list of documents they exist in.
used_words = []  # list for unique words on each line

cooccurance_cnt_dict = {} # dictionary for storing each word pair's cooccurance counts.
information_gain_dict = {} # dictionary for storing the IG value for each pair.

list_of_values = [] # a list of all the values from the dictionary 'cooccurance_cnt_dict'
list_of_IG = [] # list of Information Gain values for each pair

# getting the number of documents each word shows up in:
with open(r'C:\Users\JoshG\Documents\Information Retrieval HW Code\cacm.trec.filtered.txt','r') as f:
    
    i = 0 # used to keep track of what line we are on

    for line in f:
        i = i + 1  # new line
        for word in line.split():
            if word not in used_words:     # this is needed to ensure we dont count the same word twice on the same line
                if word not in indv_cnt_dict: 
                    indv_cnt_dict[word] = {i}    #initializing the value as a set with initial value of the line number
                else:
                    indv_cnt_dict[word].add(i)   #adding the current line (the current document) to this word's value set
                used_words.append(word)
        used_words.clear() # clearing are list of used words for the next line
    
    numOfDocs = i  # the final count of docs
    

    # getting the number of cooccurances for each pair of words:
    for pair in combinations(indv_cnt_dict.keys(), 2): 
        
        cooccurance_cnt_dict[pair] = len(indv_cnt_dict[pair[0]].intersection(indv_cnt_dict[pair[1]])) # getting the number of documents the current 2 words both show up in
        list_of_values.append(cooccurance_cnt_dict[pair])

        information_gain_dict[pair] = Information_Gain(indv_cnt_dict, cooccurance_cnt_dict,numOfDocs, pair)  # getting the information gain for the current pair
        list_of_IG.append(information_gain_dict[pair])
    



    # Problem 2.d: Top 10 Pair Occurrance Count:
    sorted_values_list = sorted(list_of_values, reverse=True)
    
    print("\nTop 10 Cooccurance Pairs:\n")

    for value in sorted_values_list[:10]:
        print(list(cooccurance_cnt_dict.keys())[list(cooccurance_cnt_dict.values()).index(value)], " : ", value)


    print("\nTop 5 Information Gain Pairs with 'programming':\n")

    #Problem 2.e: Top 10 Pair Information Gain:
    sorted_IG_list = sorted(list_of_IG, reverse=True)
    
    j = 0 # number of pairs that having 'programming' as word1 or word2
    for value in sorted_IG_list:
        pair_key = list(information_gain_dict.keys())[list(information_gain_dict.values()).index(value)]
        if(pair_key[0] == "programming" or pair_key[1] == "programming"):
            print(pair_key, " : ", value)
            j = j + 1

        if(j == 5):
            break
    
# end of program
