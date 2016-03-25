import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator
from util import contextwin


#GRU-RNN model with 2 layer of GRU taking the input as sequence of word-vector , each word-vector is constructed by window-context word embedding vector

class GRUTheano:

    def __init__(self, word_dim, window_context_size, vocab_size, num_labels, hidden_dim, bptt_truncate=-1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
	self.window_context_size=window_context_size
	self.vocab_size=vocab_size
	self.num_labels=num_labels

        # Initialize the network parameters
	# E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim)) it is for on-hot vector 
        E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (vocab_size+1, word_dim)) # word_dime here is word-embedding dimension
       
        #U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, word_dim, hidden_dim)) word_dime here is vocabulary-size
	U_1 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, word_dim*window_context_size, hidden_dim))
	U_2 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./num_labels), np.sqrt(1./num_labels), (hidden_dim,num_labels))
        b = np.zeros((6, hidden_dim))
        c = np.zeros(num_labels)

        # Theano: Created shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U_1 = theano.shared(name='U_1', value=U_1.astype(theano.config.floatX))
	self.U_2 = theano.shared(name='U_2', value=U_2.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))

        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU_1 = theano.shared(name='mU_1', value=np.zeros(U_1.shape).astype(theano.config.floatX))
	self.mU_2 = theano.shared(name='mU_2', value=np.zeros(U_2.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))


        
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

	# as many columns as context window size
        # as many lines as words in the sentence
	
    def __theano_build__(self):
        E, V, U_1, U_2, W, b, c = self.E, self.V, self.U_1, self.U_2, self.W, self.b, self.c
        
	#x = T.ivector('x')
	idxs=T.imatrix()
	x=self.E[idxs].reshape((idxs.shape[0],self.word_dim * self.window_context_size)) 
	
        y = T.ivector('y') #labels
        
        def forward_prop_step(x_t, s_t1_prev, s_t2_prev):   
      
            # GRU Layer 1
            z_t1 = T.nnet.hard_sigmoid(T.dot(x_t,self.U_1[0]) + T.dot(s_t1_prev,self.W[0]) + self.b[0])
            r_t1 = T.nnet.hard_sigmoid(T.dot(x_t,self.U_1[1]) + T.dot(s_t1_prev,self.W[1]) + self.b[1])
	    g_t1 = r_t1 * s_t1_prev
            c_t1 = T.tanh(T.dot(x_t, self.U_1[2]) + T.dot(g_t1, self.W[2]) + self.b[2])
            s_t1 = (T.ones_like(z_t1) - z_t1) *c_t1  + z_t1 * s_t1_prev
           
            # GRU Layer 2
	    
            z_t2 = T.nnet.hard_sigmoid(T.dot(s_t1,self.U_2[0]) + T.dot(s_t2_prev, self.W[3]) + self.b[3])
            r_t2 = T.nnet.hard_sigmoid(T.dot(s_t1,self.U_2[1]) + T.dot(s_t2_prev, self.W[4]) + self.b[4])
	    g_t2 = r_t2 * s_t2_prev
            c_t2 = T.tanh(T.dot(s_t1, self.U_2[2]) + T.dot(g_t2, self.W[5]) + self.b[5])
            s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev
            
            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = T.nnet.softmax(T.dot(s_t2, self.V) + self.c)

            return [o_t, s_t1, s_t2]
        
        [o, s, s2], _ = theano.scan(fn=forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,dict(initial=T.zeros(self.hidden_dim)),dict(initial=T.zeros(self.hidden_dim))],
	    n_steps=x.shape[0])
	
	p_y_given_x_sentence = o[:, 0, :]
        #y_pred
        prediction = T.argmax(p_y_given_x_sentence, axis=1)
	
	#sentence_nll
        o_error = T.sum(T.nnet.categorical_crossentropy(p_y_given_x_sentence, y))
	
        
        # Total cost (could add regularization here)
        cost = o_error
        
        
	
        # Gradients
        dE = T.grad(cost, E)
        dU_1 = T.grad(cost, U_1)
	dU_2 = T.grad(cost, U_2)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)

	# SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')
        
        # rmsprop cache updates
	
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU_1 = decay * self.mU_1 + (1 - decay) * dU_1 ** 2
	mU_2 = decay * self.mU_2 + (1 - decay) * dU_2 ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2
        
        # Assign functions
        self.predict = theano.function([idxs], o)
        self.classify = theano.function([idxs], prediction)
        self.ce_error = theano.function([idxs, y], cost)
        self.bptt = theano.function([idxs, y], [dE, dU_1, dU_2, dW, db, dV, dc])
        
        
    
	
        self.sgd_step = theano.function(
            [idxs, y, learning_rate, theano.Param(decay, default=0.9)],
            [], 
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                     (U_1, U_1 - learning_rate * dU_1 / T.sqrt(mU_1 + 1e-6)),
		     (U_2, U_2 - learning_rate * dU_2 / T.sqrt(mU_2 + 1e-6)), 	
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (self.mE, mE),
                     (self.mU_1, mU_1),
		     (self.mU_2, mU_2),	
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
                    ])

        self.normalize = theano.function(inputs=[],
                                         updates={E:
                                                  E /
                                                  T.sqrt((E**2)
                                                  .sum(axis=1))
                                                  .dimshuffle(0, 'x')})

    def train(self, x, y, window_size, learning_rate):

        cwords = contextwin(x, window_size)
        words = map(lambda x: np.asarray(x).astype('int32'), cwords)

        labels = y
	
        self.sgd_step(words, labels, learning_rate)
	self.normalize()
        
	
    def save(self, folder):
        for param in self.params:
            np.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(np.load(os.path.join(folder,
                            param.name + '.npy')))



