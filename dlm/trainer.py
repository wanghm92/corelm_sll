from __future__ import division
import theano
import theano.tensor as T
from dlm import eval
from dlm import eval_sll
import dlm.utils as U
import dlm.io.logging as L
from dlm.algorithms.lr_tuner import LRTuner
import time
import numpy as np
import sys
import time
import math


def train(classifier, criterion, args, trainset, devset, testset=None):
	if args.algorithm == "sgd":
		from dlm.algorithms.sgd import SGD as Trainer
	else:
		L.error("Invalid training algorithm: " + args.algorithm)

	# Get number of minibatches from the training file
	num_train_batches = trainset.get_num_batches()

	is_sll = args.loss_function == 'sll'
	
	# Initialize the trainer object
	trainer = Trainer(classifier, criterion, args.learning_rate, trainset, is_sll, clip_threshold=args.clip_threshold)

	# Initialize the Learning Rate tuner, which adjusts learning rate based on the development/validation file
	lr_tuner = LRTuner(low=0.01*args.learning_rate, high=10*args.learning_rate, inc=0.01*args.learning_rate)
	validation_frequency = 5000 # minibatches

	# Logging and statistics
	total_num_iter = args.num_epochs * num_train_batches
	hook = Hook(classifier, devset, testset, total_num_iter, args.out_dir, is_sll)
	L.info('Training')
	start_time = time.time()
	verbose_freq = 1000 # minibatches
	epoch = 0
	
	##########
	print_freq = 1000 #00 # minibatches
	##########

	hook.evaluate(0)
	
	a = time.time()
	classifier.save_model(args.out_dir + '/model.epoch_0.gz', zipped=True)
	
	while (epoch < args.num_epochs):
		epoch = epoch + 1
		L.info("Epoch: " + U.red(epoch))

		minibatch_avg_cost_sum = 0
		for minibatch_index in xrange(num_train_batches):
			# Makes an update of the paramters after processing the minibatch
			# minibatch_avg_cost, gparams = trainer.step(minibatch_index)
			minibatch_avg_cost ,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z, gparams = trainer.step(minibatch_index)

			minibatch_avg_cost_sum += minibatch_avg_cost

			if minibatch_index % verbose_freq == 0:
				grad_norms = [np.linalg.norm(gparam) for gparam in gparams]
				L.info(U.blue("[" + time.ctime() + "] ") + '%i/%i, current cost=%.2f, total cost=%.2f, average cost=%.2f, lr=%f'
					% (minibatch_index, num_train_batches, minibatch_avg_cost, minibatch_avg_cost_sum, minibatch_avg_cost_sum/(minibatch_index+1), trainer.get_learning_rate()))
				L.info('Grad Norms: [' + ', '.join(['%.6f' % gnorm for gnorm in grad_norms]) + ']')

				###################################################################################
				
				if minibatch_index % print_freq == 0:
					if math.isnan(minibatch_avg_cost) or math.isnan(minibatch_avg_cost_sum) or math.isnan(minibatch_avg_cost_sum/(minibatch_index+1)) :
						print "NAN"
						classifier.save_model(args.out_dir + '/model.epoch_' + str(epoch) + '_lr' + str(lr) + '_batch' + str(minibatch_index) + '.gz', zipped=True)
						sys.exit()
					print "################## minibatch_avg_cost ##################"
					print minibatch_avg_cost.shape
					print minibatch_avg_cost
					print "################## scores             ##################"
					print b.shape
					print b
					print "################## scores_roll        ##################"
					print c
					print c.shape
					print "################## transitions        ##################"
					print d
					print d.shape
					print "################## delta_0            ##################"
					print e
					print e.shape
					print "################## temp_matrix        ##################"
					print f
					print f.shape
					print "################## temp_max_matrix    ##################"
					print g
					print g.shape
					print "################## logadd_matrix      ##################"
					print h
					print h.shape
					print "################## delta_matrix       ##################"
					print i
					print i.shape
					print "################## delta_full_stack   ##################"
					print j
					print j.shape
					print "################## delta              ##################"
					print k
					print k.shape
					print "################## delta_logadd       ##################"
					print l
					print l.shape
					print "################## y_roll ##################"
					print m
					print m.shape
					print "################## y_w ##################"
					print n
					print n.shape
					print "################## y_shift ##################"
					print o
					print o.shape
					print "################## yplus1 ##################"
					print p
					print p.shape
					print "################## s ##################"
					print q
					print q.shape
					print "################## t ##################"
					print r
					print r.shape
					print "################## correct_path_score ##################"
					print s
					print s.shape					
					print "################## lookuptable ##################"
					print t
					print t.shape
					print "################## hidden layer ##################"
					print u
					print u.shape
					print "################## hidden layer -- w ##################"
					print w
					print w.shape
					print "################## hidden layer -- b ##################"
					print x
					print x.shape
					print "################## activation ##################"
					print v
					print v.shape
					print "################## scores             ##################"
					print b.shape
					print b
					print "################## linear layer -- w ##################"
					print y
					print y.shape
					print "################## linear layer -- b ##################"
					print z
					print z.shape
					print "################## gparams ##################"
					for i in xrange(len(gparams)):
						print "################## gparams %d ##################" %i
						print gparams[i]
						print gparams[i].shape
				
				###################################################################################

			curr_iter = (epoch - 1) * num_train_batches + minibatch_index
			if curr_iter > 0 and curr_iter % validation_frequency == 0:
				hook.evaluate(curr_iter)
				'''
				###################################################################################
				if False:
					print "################## minibatch_avg_cost ##################"
					print minibatch_avg_cost.shape
					print minibatch_avg_cost
					print "################## scores             ##################"
					print b.shape
					print b
					print "################## scores_roll        ##################"
					print c
					print c.shape
					print "################## transitions        ##################"
					print d
					print d.shape
					print "################## delta_0            ##################"
					print e
					print e.shape
					print "################## temp_matrix        ##################"
					print f
					print f.shape
					print "################## temp_max_matrix    ##################"
					print g
					print g.shape
					print "################## logadd_matrix      ##################"
					print h
					print h.shape
					print "################## delta_matrix       ##################"
					print i
					print i.shape
					print "################## delta_full_stack   ##################"
					print j
					print j.shape
					print "################## delta              ##################"
					print k
					print k.shape
					print "################## delta_logadd       ##################"
					print l
					print l.shape
					print "################## y_roll ##################"
					print m
					print m.shape
					print "################## y_w ##################"
					print n
					print n.shape
					print "################## y_shift ##################"
					print o
					print o.shape
					print "################## ym1 ##################"
					print p
					print p.shape
					print "################## s ##################"
					print q
					print q.shape
					print "################## t ##################"
					print r
					print r.shape
					print "################## correct_path_score ##################"
					print s
					print s.shape
					print "################## lookuptable ##################"
					print t
					print t.shape
					print "################## hidden layer ##################"
					print u
					print u.shape
					print "################## hidden layer -- w ##################"
					print w
					print w.shape
					print "################## hidden layer -- b ##################"
					print x
					print x.shape
					print "################## activation ##################"
					print v
					print v.shape
					print "################## scores             ##################"
					print b.shape
					print b
					print "################## linear layer -- w ##################"
					print y
					print y.shape
					print "################## linear layer -- b ##################"
					print z
					print z.shape
					print "################## gparams ##################"
					for i in xrange(len(gparams)):
						print "################## gparams %d ##################" %i
						print gparams[i]
						print gparams[i].shape
					
				if dev_ppl > 17:
					classifier.save_model(args.out_dir + '/model.epoch_' + str(epoch) + '_lr' + str(lr) + '_batch' + str(minibatch_index) + '.gz', zipped=True)
					sys.exit()
				###################################################################################
				'''
		L.info(U.blue("[" + time.ctime() + "] ") + '%i/%i, cost=%.2f, lr=%f'
			% (num_train_batches, num_train_batches, minibatch_avg_cost_sum/num_train_batches, trainer.get_learning_rate()))
		
		L.info("Adjusting learning rate ...")
		dev_ppl = hook.evaluate(curr_iter)
		lr = trainer.get_learning_rate()
		L.info("Learning rate is adjusted from %s" % U.red(lr))
		if args.enable_lr_adjust:
			lr = lr_tuner.adapt_lr(dev_ppl, lr)
		trainer.set_learning_rate(lr)
		L.info("...to %s" %U.red(lr))

		classifier.save_model(args.out_dir + '/model.epoch_' + str(epoch) + '.gz', zipped=True)

	end_time = time.time()
	hook.evaluate(total_num_iter)
	L.info('Optimization complete')
	L.info('Ran for %.2fm' % ((end_time - start_time) / 60.))


class Hook:
	def __init__(self, classifier, devset, testset, total_num_iter, out_dir, is_sll=False):
		self.classifier = classifier
		self.test_eval = None
		
		if is_sll:
			self.dev_eval = eval_sll.Evaluator(dataset=devset, classifier=classifier, is_sll=False)
			if testset:
				self.test_eval = eval_sll.Evaluator(dataset=testset, classifier=classifier, is_sll=False)
		else:
			self.dev_eval = eval.Evaluator(dataset=devset, classifier=classifier, is_sll=False)
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









