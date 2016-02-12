from collections import OrderedDict
import copy
import cPickle
import gzip
import os
import urllib
import random
import stat
import subprocess
import sys
import timeit

import numpy

import theano
from theano import tensor as T
from util import contextwin

class RNNSLU_LSTM(object):
    ''' elman neural net model '''
    def __init__(self, hidden_dim, num_labels, vocab_size, word_dim, window_context_size):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
        '''
	nh=hidden_dim
	nc=num_labels
	ne=vocab_size
	de=word_dim
	cs=window_context_size
        # parameters of the model
        self.emb = theano.shared(name='embeddings',
                                 value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                 (ne+1, de))
                                 # add one for padding at the end
                                 .astype(theano.config.floatX))
        self.wx_i = theano.shared(name='wx_i',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.wx_f = theano.shared(name='wx_f',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.wx_o = theano.shared(name='wx_o',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.wx_ctilda = theano.shared(name='wx_ctilda',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))   
                                                          
        self.wh_i = theano.shared(name='wh_i',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.wh_f = theano.shared(name='wh_f',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))     
        self.wh_o = theano.shared(name='wh_o',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.wh_ctilda = theano.shared(name='wh_ctilda',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))                                                          
                                
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, nc))
                               .astype(theano.config.floatX))
                               
        self.bh_i = theano.shared(name='bh_i',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
                                
        self.bh_f = theano.shared(name='bh_f',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
                                
        self.bh_o = theano.shared(name='bh_o',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))                                              
                                
        self.bh_ctilda = theano.shared(name='bh_ctilda',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))  
                                                     
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nc,
                               dtype=theano.config.floatX))
       
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.c0 = theano.shared(name='c0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
       

        # bundle
        self.params = [self.emb, self.wx_i,self.wx_f,self.wx_o,self.wx_ctilda,
                        self.wh_i,self.wh_f,self.wh_o,self.wh_ctilda,
                        self.w,
                        self.bh_i,self.bh_f,self.bh_o,self.bh_ctilda,
                        self.b, self.h0, self.c0]
        # end-snippet-2
        # as many columns as context window size
        # as many lines as words in the sentence
        # start-snippet-3
        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y_sentence = T.ivector('y_sentence')  # labels
        # end-snippet-3 start-snippet-4

        def recurrence(x_t,c_tm1,h_tm1):
            #forget
            f_t = T.nnet.hard_sigmoid(T.dot(x_t, self.wx_f) + T.dot(h_tm1, self.wh_f) + self.bh_f)
            #input
            i_t = T.nnet.hard_sigmoid(T.dot(x_t, self.wx_i) + T.dot(h_tm1, self.wh_i) + self.bh_i)
           
           #output
            ctilda_t = T.tanh(T.dot(x_t, self.wx_ctilda) + T.dot(h_tm1, self.wh_ctilda) + self.bh_ctilda)
            o_t = T.nnet.hard_sigmoid(T.dot(x_t, self.wx_o) + T.dot(h_tm1, self.wh_o) + self.bh_o)
            
            c_t = f_t * c_tm1 + i_t * ctilda_t
            h_t =  o_t * T.tanh(c_t) 
            
            # Final output calculation
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            
            return [h_t, c_t, s_t]

        [h, c, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
				truncate_gradient=-1,
                                outputs_info=[self.h0, self.c0, None],
                                n_steps=x.shape[0])

        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)
        
        # end-snippet-4

        # cost and gradients and learning rate
        # start-snippet-5
        lr = T.scalar('lr')
        
        #sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               #[T.arange(x.shape[0]), y_sentence])
        sentence_nll=T.sum(T.nnet.categorical_crossentropy(p_y_given_x_sentence,y_sentence))
        sentence_gradients = T.grad(sentence_nll, self.params)
        
        sentence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))
        # end-snippet-5

        # theano functions to compile
        # start-snippet-6
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)
        # end-snippet-6 start-snippet-7
        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:
                                                  self.emb /
                                                  T.sqrt((self.emb**2)
                                                  .sum(axis=1))
                                                  .dimshuffle(0, 'x')})
        # end-snippet-7

    def train(self, x, y, window_size, learning_rate):

        cwords = contextwin(x, window_size)
        words = map(lambda x: numpy.asarray(x).astype('int32'), cwords)
        labels = y

        self.sentence_train(words, labels, learning_rate)
        self.normalize()

    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))
