import nltk,re
from nltk import word_tokenize 
import itertools
import numpy as np
import time
import sys
import operator
import io
import os
import array
from six.moves import cPickle as pickle
from datetime import datetime
from gru import GRUTheano
from collections import OrderedDict
import copy
import gzip
import urllib
import random
import stat
import subprocess
import timeit
import theano
from theano import tensor as T

path='/user/fnoorala/home/Desktop/SMILK/InformationExtraction/data/SemVAl/'

SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
UNKNOWN_TOKEN = "<UNK>"
word_to_index = []
index_to_word = []
vocabulary_size=8000

def readfiles(filename):
     with open(filename, 'r') as f:
        lines = [line.rstrip('\n') for line in open(filename) 
                 if (len(line.strip())>0 and
                 not line.lower().startswith('comment:'))]
        return lines
    
# find the nominals in the phrase

def extractNominals(mystr):
        nominals=[ match.group(1) for match in re.finditer(r'(<e\d{1}>.*?</e\d{1}>)',mystr)]
        return nominals

def convertToDigit(x):
        t=''
        for i in range(len(x)):
            t += 'DIGIT'
        return t       
# find the order of the nominal in relation if 1 means (e1,e2) if 2 means (e2,e1) and if 0 means 
# no any relation
def extractOrders(s):
    matchobj=re.search('(.*?)\((.*?)\)',s)
    order=0
    relation='other'
    if matchobj:
        order=1 if matchobj.group(2).split(',')[0]=='e1' else 2
        relation=matchobj.group(1).split('-')     
    return {'rel':relation, 'order':order}

def transformation(filename,flag):
    dataset=readfiles(filename)
    texts=[]
    tags=[]
    for k in range(0,len(dataset),2):
        result_d = {}
        d=extractOrders(dataset[k+1])
        text=dataset[k]
        if d['order']==1:
            result_d[extractNominals(text)[0]]=d['rel'][0]
            result_d[extractNominals(text)[1]]=d['rel'][1]
        elif d['order']==2:
            result_d[extractNominals(text)[0]]=d['rel'][1]
            result_d[extractNominals(text)[1]]=d['rel'][0]
        else:
            result_d[extractNominals(text)[0]]='Other_1'
            result_d[extractNominals(text)[1]]='Other_2'
        for key, value in result_d.iteritems():
            
            nominal=re.sub('<.*?>','',key)
            if len(nominal.split(" ")) < 2:
                result_d[key]="B+"+value
            else:
                temp="B+"+value
                for j in range(len(re.findall(r'(\S\s)',nominal))):
                    temp=temp+" "+"I+"+value
                    result_d[key]=temp

        # use these three lines to do the replacement
        result_d = dict((re.escape(k), v) for k, v in result_d.iteritems())

        pattern = re.compile("|".join(result_d.keys()))

        text=pattern.sub(lambda m: result_d[re.escape(m.group(0))], text)
        t=dataset[k]
        
        if flag:
            text=re.sub(r'^(\d+)(\s)',r'\1\t',text)
            t=re.sub(r'^(\d+)(\s)',r'\1\t',t)
            text=text.split('\t')[1][1:-1]
            #delete tags and id number for each sentence
            t= re.sub('<.*?>','',t.split('\t')[1][1:-1])
        else:    
            text=text.split('\t')[1][1:-2]
            t= re.sub('<.*?>','',t.split('\t')[1][1:-2])
            
        tokens=nltk.word_tokenize(text)
        rg = re.compile('[BI]\+(?:[A-Za-z][a-z0-9_]*)')
                
        #we converted sequences of numbers with the string DIGIT i.e. 1984
        #is converted to DIGITDIGITDIGITDIGIT.
        #t=re.sub(r"\d+",lambda m:convertToDigit(m.group()),t)
        
        tokens=[ x if rg.match(x) is not None else 'O' for x in tokens ]
        texts.append(t.decode("utf-8").lower())
        tags.append(tokens)
        
        
        
    return texts,tags

def dataset(path):
    
    train='SemEval2010_task8_data_release/TRAIN_FILE.TXT'
    test='semeval_trial_data/trial_data.shuffled'
    data={'train':path+train, 'test':path+test}
    
    train_set,train_labels=transformation(data['train'],False)
    test_set,test_labels=transformation(data['test'],True)
      
    
    #sentences = ['%s %s %s' % (SENTENCE_START_TOKEN, x, SENTENCE_END_TOKEN) for x in sentences]
    #labels=[[SENTENCE_START_TOKEN]+l+[SENTENCE_END_TOKEN] for l in labels]

    # Tokenize the sentences into words
    
    train_tokenized_sentences = [nltk.word_tokenize(sent) for sent in train_set]
    
    #we converted sequences of numbers with the string DIGIT i.e. 1984
    #is converted to DIGITDIGITDIGITDIGIT.
  
    train_tokenized_sentences=[[re.sub(r"\d+",lambda m:convertToDigit(m.group()),w) for w in t ] for t in train_tokenized_sentences]
    
    test_tokenized_sentences=[nltk.word_tokenize(sent) for sent in test_set]
    
    test_tokenized_sentences=[[re.sub(r"\d+",lambda m:convertToDigit(m.group()),w) for w in t ] for t in test_tokenized_sentences]
    
    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*train_tokenized_sentences))
    
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)

    #we deal with unseen words in the test set by marking any words with only one 
    #single occurrence in the training set as <UNK>
    vocab = [x for x in vocab if not x[1]<2]

    #print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))
    sorted_vocab = sorted(vocab, key=operator.itemgetter(1))
    index_to_word = ["<MASK/>", UNKNOWN_TOKEN] + [x[0] for x in sorted_vocab]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

 
    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(train_tokenized_sentences):
        train_tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]
    for i, sent in enumerate(test_tokenized_sentences):
        test_tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

    index_to_label = set([item for sublist in train_labels for item in sublist])
    label_to_index = dict([(w, i) for i, w in enumerate(index_to_label)])
    
    dicts={'words2idx':word_to_index,'labels2idx':label_to_index}
    # Create the training dataset and test dataset
    X_train = np.asarray([[word_to_index[w] for w in sent] for sent in train_tokenized_sentences])
    y_train = np.asarray([[label_to_index[w] for w in sent] for sent in train_labels])
    
    
    
    X_test = np.asarray([[word_to_index[w] for w in sent] for sent in test_tokenized_sentences])
    y_test = np.asarray([[label_to_index[w] for w in sent] for sent in test_labels])
    
    
    semeval={'train_dataset':X_train,'train_labels':y_train,'test_dataset':X_test,
             'test_labels':y_test,'dicts':dicts}
    return semeval
#save the train, test and dictionary in a serialized object
pickle.dump( dataset(path), open( "semeval.pkl", "wb" ) )
