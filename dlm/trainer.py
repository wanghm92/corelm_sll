from __future__ import division
import theano
import theano.tensor as T
from dlm import eval
import dlm.utils as U
import dlm.io.logging as L
from dlm.algorithms.lr_tuner import LRTuner
import time
import numpy as np
import sys
import time


def train(classifier, criterion, args, trainset, devset, testset=None):
	if args.algorithm == "sgd":
		from dlm.algorithms.sgd import SGD as Trainer
	else:
		L.error("Invalid training algorithm: " + args.algorithm)

	# Get number of minibatches from the training file
	num_train_batches = trainset.get_num_batches()

	is_sll = False
	if args.loss_function == 'sll':
		is_sll = True
	
	is_sll=True

	# Initialize the trainer object
	trainer = Trainer(classifier, criterion, args.learning_rate, trainset, is_sll, clip_threshold=args.clip_threshold)

	# Initialize the Learning Rate tuner, which adjusts learning rate based on the development/validation file
	lr_tuner = LRTuner(low=0.01*args.learning_rate, high=10*args.learning_rate, inc=0.01*args.learning_rate)
	validation_frequency = 5000 # minibatches

	# Logging and statistics
	total_num_iter = args.num_epochs * num_train_batches
	hook = Hook(classifier, devset, testset, total_num_iter, args.out_dir)
	L.info('Training')
	start_time = time.time()
	verbose_freq = 1000 # minibatches
	epoch = 0
	
	hook.evaluate(0)
	
	a = time.time()
	classifier.save_model(args.out_dir + '/model.epoch_0.gz', zipped=True)
	
	while (epoch < args.num_epochs):
		epoch = epoch + 1
		L.info("Epoch: " + U.red(epoch))

		minibatch_avg_cost_sum = 0
		for minibatch_index in xrange(num_train_batches):
			# Makes an update of the paramters after processing the minibatch
			minibatch_avg_cost, gparams = trainer.step(minibatch_index)
			minibatch_avg_cost_sum += minibatch_avg_cost
			
			if minibatch_index % verbose_freq == 0:
				grad_norms = [np.linalg.norm(gparam) for gparam in gparams]
				L.info(U.blue("[" + time.ctime() + "] ") + '%i/%i, cost=%.2f, lr=%f'
					% (minibatch_index, num_train_batches, minibatch_avg_cost_sum/(minibatch_index+1), trainer.get_learning_rate()))
				L.info('Grad Norms: [' + ', '.join(['%.6f' % gnorm for gnorm in grad_norms]) + ']')
			curr_iter = (epoch - 1) * num_train_batches + minibatch_index
			if curr_iter > 0 and curr_iter % validation_frequency == 0:
				hook.evaluate(curr_iter)

		L.info(U.blue("[" + time.ctime() + "] ") + '%i/%i, cost=%.2f, lr=%f'
			% (num_train_batches, num_train_batches, minibatch_avg_cost_sum/num_train_batches, trainer.get_learning_rate()))
		dev_ppl = hook.evaluate(curr_iter)
		lr = trainer.get_learning_rate()
		if args.enable_lr_adjust:
			lr = lr_tuner.adapt_lr(dev_ppl, lr)
		trainer.set_learning_rate(lr)
		classifier.save_model(args.out_dir + '/model.epoch_' + str(epoch) + '.gz', zipped=True)

	end_time = time.time()
	hook.evaluate(total_num_iter)
	L.info('Optimization complete')
	L.info('Ran for %.2fm' % ((end_time - start_time) / 60.))


class Hook:
	def __init__(self, classifier, devset, testset, total_num_iter, out_dir, is_sll=False):
		self.classifier = classifier
		self.dev_eval = eval.Evaluator(dataset=devset, classifier=classifier, is_sll=False)
		self.test_eval = None
		if testset:
			self.test_eval = eval.Evaluator(dataset=testset, classifier=classifier, is_sll=False)
		self.best_iter = 0
		self.best_dev_perplexity = np.inf
		self.best_test_perplexity = np.inf
		self.t0 = time.time()
		self.total_num_iter = total_num_iter
		self.out_dir = out_dir

	def evaluate(self, curr_iter):
		denominator = self.dev_eval.get_denominator()
		dev_error = self.dev_eval.classification_error()
		dev_perplexity = self.dev_eval.perplexity()
		if self.test_eval:
			test_error = self.test_eval.classification_error()
			test_perplexity = self.test_eval.perplexity()

		if dev_perplexity < self.best_dev_perplexity:
			self.best_dev_perplexity = dev_perplexity
			self.best_iter = curr_iter
			if self.test_eval:
				self.best_test_perplexity = test_perplexity

		if curr_iter > 0:
			t1 = time.time()
			rem_time = int((self.total_num_iter - curr_iter) * (t1 - self.t0) / (curr_iter * 60))
			rem_time = str(rem_time) + "m"
		else:
			rem_time = ""

		L.info(('DEV  => Error=%.2f%%, PPL=' + U.b_yellow('%.2f @ %i') + ' (' + U.b_red('%.2f @ %i') + '), Denom=%.3f, %s')
			% (dev_error * 100., dev_perplexity, curr_iter, self.best_dev_perplexity, self.best_iter, denominator, rem_time))
		if self.test_eval:
			L.info(('TEST => Error=%.2f%%, PPL=' + U.b_yellow('%.2f @ %i') + ' (' + U.b_red('%.2f @ %i') + ')')
				% (test_error * 100., test_perplexity, curr_iter, self.best_test_perplexity, self.best_iter))
		
		return dev_perplexity









