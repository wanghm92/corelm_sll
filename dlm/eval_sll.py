from __future__ import division
import sys
import time
from theano import *
import theano.tensor as T
from dlm.io.mmapReader import MemMapReader
from dlm.models.mlp import MLP
import dlm.utils as U
import math
import numpy as np

class Evaluator():

	def __init__(self, dataset, classifier, is_sll):

		index = T.lscalar()
		x = classifier.input
		y = T.ivector('y')

		self.dataset = dataset								# Initializing the dataset
		self.num_batches = self.dataset.get_num_batches()	# Number of minibatches in the dataset
		self.num_samples = self.dataset._get_num_samples()	# Number of samples in the dataset

		# --------------------  viterbi  -------------------- #

		self.transitions = classifier.A
		self.scores = classifier.score_output()

		self.delta_0 = self.transitions[0] + self.scores[0]
			
		self.transitions_tranc = self.transitions[1:].T # A_k,i
		self.scores_roll = T.roll(self.scores, -self.scores.shape[1])

		def calculate_delta_recursively(net_scores, delta_prev):
			temp = delta_prev + self.transitions_tranc
			temp_max = T.max(temp, axis=1)
			graph = T.argmax(temp, axis=1)
			delta = net_scores + temp_max

			return temp, temp_max, graph, delta

		[temp, temp_max, graph, delta], updates = theano.scan(
			calculate_delta_recursively,
			sequences=[self.scores_roll],
			outputs_info=[None,None,None, dict(initial=self.delta_0)])

		self.temp_matrix = temp
		self.temp_max_matrix = temp_max

		self.delta_flat = T.flatten(delta)
		self.delta_full_flat = T.concatenate([self.delta_0, self.delta_flat],axis=0)
		self.delta_full_stack = T.reshape(self.delta_full_flat,(delta.shape[0]+1,delta.shape[1]))
		self.delta = self.delta_full_stack[:-1]
		
		# self.graph = graph
		self.graph_0 = T.arange(graph.shape[1])
		self.graph_flat = T.flatten(graph)
		self.graph_full_flat = T.concatenate([self.graph_0, self.graph_flat],axis=0)
		self.graph_full_stack = T.reshape(self.graph_full_flat,(graph.shape[0]+1,graph.shape[1]))
		self.graph = self.graph_full_stack[:-1]
		
		self.path_end = T.max(self.delta[-1])
		self.path_end_index = T.argmax(self.delta[-1])

		def find_path(step, last_index):
			idx = step[last_index]
			return idx

		path,updates = theano.scan(
			find_path,
			sequences=self.graph,
			outputs_info=[dict(initial=self.path_end_index)],
			go_backwards=True)
		
		self.path_before = path

		self.path = T.concatenate([[self.path_end_index], path],axis=0)[::-1][1:]
		
		self.get_y_pred = theano.function(
			inputs=[index],
			outputs=self.path,
			givens={
				x:self.dataset.get_x(index,is_sll)
			}
		)
		
		self.test = theano.function(
			inputs=[index],
			outputs=[self.transitions, self.scores, self.delta_0, self.transitions_tranc, self.scores_roll, self.temp_matrix, self.temp_max_matrix, self.delta_full_stack, self.delta, self.graph, self.path_end, self.path_end_index, self.path, self.path_before],
			givens={
				x:self.dataset.get_x(index,is_sll)
			}
		)
	
		# --------------------  end viterbi  -------------------- #

		self.denominator = theano.function(
			inputs=[index],
			outputs=classifier.log_Z_sqr,
			givens={
				x: self.dataset.get_x(index,is_sll)
			}
		)

		self.sum_batch_error = theano.function(
			inputs=[index],
			outputs=self.errors(y),
			givens={
				x: self.dataset.get_x(index,is_sll),
				y: self.dataset.get_y(index,is_sll)
			}
		)

		self.neg_sum_batch_log_likelihood = theano.function(
			inputs=[index],
			outputs=-T.sum(T.log(classifier.p_y_given_x(y))),
			givens={
				x: self.dataset.get_x(index,is_sll),
				y: self.dataset.get_y(index,is_sll)
			}
		)

		self.neg_sequence_log_prob = self.neg_sum_batch_log_likelihood

		self.unnormalized_neg_sum_batch_log_likelihood = theano.function(
			inputs=[index],
			outputs=-T.sum(classifier.unnormalized_p_y_given_x(y)), # which is: -T.sum(T.log(T.exp(classifier.unnormalized_p_y_given_x(y))))
			givens={
				x: self.dataset.get_x(index,is_sll),
				y: self.dataset.get_y(index,is_sll)
			}
		)
		
	def errors(self, y):
		if y.ndim != self.path.ndim:
			raise TypeError('y should have the same shape as self.path', ('y', y.type, 'path', self.path.type))
		if y.dtype.startswith('int'):
			return T.sum(T.neq(self.path, y))
		else:
			raise NotImplementedError()
	
	def get_batch_predicted_class(self, index):
		return self.get_y_pred(index)
		# return self.test(index)

	def get_denominator(self):
		return np.mean([self.denominator(i) for i in xrange(self.num_batches)])

	def classification_error(self):
		return np.sum([self.sum_batch_error(i) for i in xrange(self.num_batches)]) / self.num_samples
		
	def mean_neg_log_likelihood(self):
		return math.fsum([self.neg_sum_batch_log_likelihood(i) for i in xrange(self.num_batches)]) / self.num_samples # np.sum() has some precision problems here

	def perplexity(self):
		return math.exp(self.mean_neg_log_likelihood())

	def mean_unnormalized_neg_log_likelihood(self):
		return math.fsum([self.unnormalized_neg_sum_batch_log_likelihood(i) for i in xrange(self.num_batches)]) / self.num_samples # np.sum() has some precision problems here

	def unnormalized_perplexity(self):
		return math.exp(self.mean_unnormalized_neg_log_likelihood())
	
	def get_sequence_log_prob(self, index):
		return - self.neg_sequence_log_prob(index)

	def get_unnormalized_sequence_log_prob(self, index):
		return - self.unnormalized_neg_sum_batch_log_likelihood(index)

		'''
		# x: A matrix (N * (ngram - 1)) representing the sequence of length N
		# y: A vector of class labels


		self.get_p_matrix  = theano.function(
			inputs=[index],
			outputs=classifier.p_y_given_x_matrix,
			givens={
				x:self.dataset.get_x(index,is_sll)
			}
		)
		
		# End of if

		self.ngram_log_prob = theano.function(
			inputs=[x, y],
			outputs=T.log(classifier.p_y_given_x(y)),
		)

	def get_ngram_log_prob(self, x, y):
		return self.ngram_log_prob(x, y)

	def get_class(self, index, restricted_ids = []):
		if restricted_ids != []:
			return restricted_ids[np.argmax(self.get_p_matrix(index)[:,restricted_ids])]
		else:
			return self.get_y_pred(index)[0]

	'''