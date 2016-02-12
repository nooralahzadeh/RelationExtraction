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
from lstm import RNNSLU_LSTM
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
from util import shuffle, minibatch, contextwin
from accuracy import conlleval

# Otherwise the deepcopy fails
import sys
sys.setrecursionlimit(15000)

# In[4]:

def main(param=None):
	if not param:
		param = {'lr': 0.0970806646812754,
		    'verbose': 1,
		    'decay': True,
		    # decay on the learning rate if improvement stops
		    'win': 3,
		    # number of words in the context window
		    'nhidden': 200,
		    # number of hidden units
		    'seed': 345,
		    'emb_dimension': 50,
		    # dimension of word embedding
		    'nepochs': 60,
		    # 60 is recommended
		    'savemodel': False}
	print param

	folder = "RelationExtraction"
	if not os.path.exists(folder):
		os.mkdir(folder)
	#load dataset
	pickle_file = 'semeval.pkl'
	with open(pickle_file, 'rb') as f:
	    save = pickle.load(f)
	    train_dataset = save['train_dataset']
	    train_labels = save['train_labels']
	    test_dataset = save['test_dataset']
	    test_labels = save['test_labels']
	    dic=save['dicts']
	    del save  # hint to help gc free up memory  
	    print('Training set', train_dataset.shape, train_labels.shape)
	    print('Test set', test_dataset.shape, test_labels.shape)


	# In[5]:
	train_dataset=[np.array(x,dtype=np.int32) for x in train_dataset]
	train_labels=[np.array(x,dtype=np.int32) for x in train_labels]
	x_test=[np.array(x,dtype=np.int32) for x in test_dataset]
	y_test=[np.array(x,dtype=np.int32) for x in test_labels]

	x_train=train_dataset[0:6000]
	y_train=train_labels[0:6000]
	x_valid=train_dataset[6001:8000]
	y_valid=train_labels[6001:8000]
	
	

	# In[6]:

	#Raw input encoding -''' visualize a few sentences '''
	w2idx,labels2idx = dic['words2idx'], dic['labels2idx']
	idx2w  = dict((v,k) for k,v in w2idx.iteritems())
	idx2la = dict((v,k) for k,v in labels2idx.iteritems())  

	# In[10]:

	vocsize = len(idx2w)
	nclasses = len(idx2la)
	nsentences = len(x_train)

	groundtruth_valid = [map(lambda x: idx2la[x], y) for y in y_valid]
	words_valid = [map(lambda x: idx2w[x], w) for w in x_valid]
	groundtruth_test = [map(lambda x: idx2la[x], y) for y in y_test]
	words_test = [map(lambda x: idx2w[x], w) for w in x_test]

	# instanciate the model
	np.random.seed(param['seed'])
	random.seed(param['seed'])
 
	rnn = GRUTheano(word_dim=param['emb_dimension'], window_context_size=param['win'], vocab_size=vocsize, num_labels=nclasses, hidden_dim=param['nhidden'])
	#rnn = RNNSLU_LSTM(hidden_dim=param['nhidden'], num_labels=nclasses, vocab_size=vocsize, word_dim=param['emb_dimension'], window_context_size=param['win'])

	# train with early stopping on validation set
	best_f1 = -np.inf
    	param['clr'] = param['lr']
    	for e in xrange(param['nepochs']):

		# shuffle
		shuffle([x_train, y_train], param['seed'])

		param['ce'] = e
		tic = timeit.default_timer()
		
		for i, (x, y) in enumerate(zip(x_train, y_train)):
		    rnn.train(x, y, param['win'], param['clr'])
		    print '[learning] epoch %i >> %2.2f%%' % (
		        e, (i + 1) * 100. / nsentences),
		    print 'completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic),
		    sys.stdout.flush()

		# evaluation // back into the real world : idx -> words
		predictions_test = [map(lambda x: idx2la[x],
		                    rnn.classify(np.asarray(
		                    contextwin(x, param['win'])).astype('int32')))
		                    for x in x_test]
		predictions_valid = [map(lambda x: idx2la[x],
		                     rnn.classify(np.asarray(
		                     contextwin(x, param['win'])).astype('int32')))
		                     for x in x_valid]

		# evaluation // compute the accuracy using conlleval.pl
		res_test = conlleval(predictions_test,
		                     groundtruth_test,
		                     words_test,
		                     folder + '/current.test.txt',
		                     folder)
		res_valid = conlleval(predictions_valid,
		                      groundtruth_valid,
		                      words_valid,
		                      folder + '/current.valid.txt',
		                      folder)

		if res_valid['f1'] > best_f1:

		    if param['savemodel']:
		        rnn.save(folder)

		    best_rnn = copy.deepcopy(rnn)
		    best_f1 = res_valid['f1']

		    if param['verbose']:
		        print('NEW BEST: epoch', e,
		              'valid F1', res_valid['f1'],
		              'best test F1', res_test['f1'])

		    param['vf1'], param['tf1'] = res_valid['f1'], res_test['f1']
		    param['vp'], param['tp'] = res_valid['p'], res_test['p']
		    param['vr'], param['tr'] = res_valid['r'], res_test['r']
		    param['be'] = e

		    subprocess.call(['mv', folder + '/current.test.txt',
		                    folder + '/best.test.txt'])
		    subprocess.call(['mv', folder + '/current.valid.txt',
		                    folder + '/best.valid.txt'])
		else:
		    if param['verbose']:
		        print ''

		# learning rate decay if no improvement in 10 epochs
		if param['decay'] and abs(param['be']-param['ce']) >= 10:
		    param['clr'] *= 0.5
		    rnn = best_rnn

		if param['clr'] < 1e-5:
		    break

	print('BEST RESULT: epoch', param['be'],
		  'valid F1', param['vf1'],
		  'best test F1', param['tf1'],
		  'with the model', folder)


if __name__ == '__main__':
    main()
