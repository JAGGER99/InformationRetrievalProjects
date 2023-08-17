
# author: Josh Greene 
# date: 9/11/22
# class: Information Retrieval
# Prof: Dr. Santu
# HW #3 part 4: TF-IDF Heuristic


import csv
import math
import string
import sys


csv.field_size_limit(int(sys.maxsize/10000000000)) # increasing the max size of the 'speech' field from the csv file since there are some rows where the text is very large. 


with open(r'Documents\Information Retrieval HW Code\state-of-the-union.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    TF_dict = {} # key: word , value: is list of counts for document 'doc_count' where 'doc_count' is the index of the list...   key-value pair example:  'the' : [doc1 count, doc2 count, doc3 count, doc4 count,...]
    zero_list = [] # a list for having a default value of zero for every word through out all the documents that will get overwritten once we actually find that word for the first time.
    doc_count = 0 # the document we are on
    year_speech_num_dict = {} # dictionary where key = decade : value = number of speeches in decade
    decades = [1790,1800,1810,1820,1830,1840,1850,1860,1870,1880,1890,1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020]
    i = 0 # index for decades

    # now actually splitting up the csv file collecting:
    for row in csv_reader:

        #adding another index-space for the new document
        for key in TF_dict:
            TF_dict[key].append(0)

        zero_list.append(0) # used for words we haven't seen yet

        year = row[0]
        speech = row[1]

        # creating the speeches per decade dictionary:
        if (int(year) not in range(decades[i], decades[i+1])):
            i += 1
        if decades[i] not in year_speech_num_dict:
            year_speech_num_dict[decades[i]] = 1
        else:
            year_speech_num_dict[decades[i]] += 1
            

        
        speech = speech.lower()  # making all words in the string 'speech' lower case
        speech = speech.translate(str.maketrans('', '', string.punctuation)) # removing all punctuation characters
        speech_word_list = speech.split() # making a list of words split on white space


        for word in speech_word_list:

            if word not in TF_dict:
               TF_dict[word] = zero_list.copy()

            TF_dict[word][doc_count] += 1

        doc_count += 1

 
    ################################################################################################################################

    total_doc_count_dict = {}  # key: word , value: total number of documents the word exists at least once in
    
    for word in TF_dict:
        
        total_doc_count_dict[word] = 0

        for count in TF_dict[word]:

            if (count > 0):    # <-- if the word was found at least once in this document

                total_doc_count_dict[word] += 1
    
    
    ################################################################################################################################
    
    IDF_weights_dict = {} # the IDF weight for each word

    for word in total_doc_count_dict:

        IDF_weights_dict[word] = math.log2(doc_count / total_doc_count_dict[word])
        
    #################################################################################################################################
    # multiplying the TF vectors for each document by the IDF weights for each term:

    TF_IDF_dict = {} # key = word : value = [ doc1 count, doc2 count, doc3 count, ...]
    total_per_doc = {}  # key = doc number , value = the sum of every word's TF_IDF for this document ...to be used for normalization


    for i in range(doc_count):
        
        total_per_doc[i] = 0

        for word in TF_dict:

            if word not in TF_IDF_dict:
                TF_IDF_dict[word] = []

            TF_IDF_dict[word].append(TF_dict[word][i] * IDF_weights_dict[word])
    
            total_per_doc[i] += TF_IDF_dict[word][i]


    ###################################################################################################################################
    #normalization:
    
    normalized_TF_IDF_vector_per_speech = {}  # key = doc number , value = dictionary of { key = word , value = TF_IDF_dict[word][i] / total_per_doc[i] } pairs

    for i in range(doc_count):
        
        normalized_TF_IDF_vector_per_speech[i] = {}
        
        for word in TF_dict:
            
            if (TF_IDF_dict[word][i] > 0):   # only care about the words that are in this document

                normalized_TF_IDF_vector_per_speech[i].update({word : (TF_IDF_dict[word][i] / total_per_doc[i]) })
            
    
    #################################################################################################################################
    
    # I randomly chose speech #81:

    dict = normalized_TF_IDF_vector_per_speech[80]
    
    print('\nTop 20 terms in Union Speech 81:\n')

    for w in sorted(dict, key=dict.get, reverse=True)[:20]:
        print(w, dict[w])


    """
    Commentary:

    Looking at the 20 words that are provided, it would seem to be that this speech was given by Ulysses S. Grant since we see the '1869' and '1868'.
    I believe, because of the IDF factor of our weights, it is safe to assume that these are representing the year the speech was given.
    Another interesting insight would be the words 'spain' and 'cuba' and 'belligerency'. If I'm remembering my history class correctly, this was close to the Spanish-American War
    which started in 1898. So, the president was probably addressing some early signs of this conflict. Another interesting group of words are 'cent' , 'currency' , 'bonds' and 'reconstruction'.
    My guess about why these are present is because this is right after the American Civil War and thus the South would have been completely devastated economically,
    and the president was probably addressing this and discussing how to resolve those issues with the South coming back into the US. The word 'coolies' is an interesting one,
    that I hadn't heard of before, but after looking it up, it might be here because there was some event concerning the Chinese coolies who were working on the railroads over the
    last 30 years. This was around the time when America was building railroads from coast to coast, maybe it was something concerning that???

    """
    #################################################################################################################################
    
    decade_vector_dict = {}  # key = the decade  :  value = a dictionary of { key = word : value = sum of each speech in the decade's 'normalized_TF_IDF_vector_per_speech' for word  } pairs
    doc_start = 0   # the document number to start on for the current decade
    doc_sum = 0  # the cumulative document count as we move through the decades
    
    for decade in year_speech_num_dict:
        
        decade_vector_dict[decade] = {}
        doc_sum += year_speech_num_dict[decade]

        for i in range(doc_start, doc_sum):
            
            for word in normalized_TF_IDF_vector_per_speech[i]:

                if (word not in decade_vector_dict[decade]):
                    decade_vector_dict[decade].update({word : normalized_TF_IDF_vector_per_speech[i][word]})
                else:
                    decade_vector_dict[decade][word] += normalized_TF_IDF_vector_per_speech[i][word]
        
        doc_start = doc_sum
        
    ###################################################################################################################################
    #printing the top 20 terms for each decade:

    for decade in decade_vector_dict:
        print("\nNEW DECADE : ", decade, ":\n")
        for w in sorted(decade_vector_dict[decade], key=decade_vector_dict[decade].get, reverse=True)[:20]:
            print(w, decade_vector_dict[decade][w])
            
    """ 
    Commentary:

    
    1900's: During the 1900's, America was clearing out forest and building the Panama Canal for trading purposes. Tariffs were being applied to imported and exported goods to help America's booming economy.
            There also seems to be the development of businesses and corporations where people could earn wages for work. We must have also been in some kind of relation with the phillippine
            islands. It could have something to do with trade or possibly a war.


    1910's: During the 1910's, America seemed to be continuing its industrial revolution and building of railroads for people to travel from state to state. We were also involved with Germans during
            this time. With the word 'german' along with the word 'unrest', I would assume it is due to the fact that World War I was occuring at the time. It also looks like Woodrow Wilson was president
            during this time.


    1920's: During the 1920's, America was experiencing major economic growth. After World War I, there was an agriculture issue, where US farmers were having a difficult time producing and keeping up
            with demand. This would be the start of the Great Depression.


    1930's: During the 1930's, America was going through the worst economic state of its entire life, The Great Depression. There was massive economic uncertainty and unemployment rates were through
            the roof. However, towards the end of the decade, America was able to slowly pull itself out of the depression and begin recovery.


    1940's: During the 1940's, America would part-take in World War II. After the japanese bombed Pearl Harbor, we decided to enter the fight against Hitler's Nazi Germany along with the Japanese. 
            Millions of soldiers lost their lives protecting this country. America would start rebuilding in the later half of the decade.
            


    1950's: During the 1950's, America would begin its fight against the Soviet communist party. This decade would be the start of the production of atomic weapons by the US and Russia.
            We also entered conflict with Korea during the Korean War. 


    1960's: During the 1960's, America was still fighting against the Sovient communist party, constantly threatening nuclear war with each other. We also started another war with Vietnam. 


    1970's: During the 1970's, America was doing a lot of spending and consequently produced inflation in the country. The country was also in a huge energy crisis, and as a result, the governement
            created the Department of Enegery I assume to help combat this and prevent it from happening again.


    1980's: During the 1980's, America was still competeing with the soviets in the nuclear space. America also started to help fight in Afghanistan against the soviets who had invaded it.
            We also were trying to maintain our economy, because of the debt we were starting to accumulate through all our war expenses.


    1990's: During the 1990's, there was a shift in the traditions of family life that had been present in previous generations. With the introduction of new technology, children were more and more
            distant from their parents. There was also more emphasis put on going to college and getting a degree in order to secure a job. 


    2000's: During the 2000's, America would begin a new war with Iraqi Terrorists lead by al qeda and saddam hussein. Medicare was also introduced with the intention of free health care for everyone.
            There was also another economic crash in 2008, though this one was not nearly as severe as the great depression in the 1920's.


    2010's: During the 2010's, America is continuing to place great importance in the education of the american youth. People are continually encouraged to go to college to help discover new
            innovations and get jobs to improve the economy. 

    """