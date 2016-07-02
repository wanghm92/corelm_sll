#!/usr/bin/env python

import sys
import time
import argparse
import dlm.utils as U
import dlm.io.logging as L
from dlm.io.vocabReader import VocabManager
###############
## Arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test-file", dest="test_path", required=True, help="The evaluation file (memory-mapped, nbest list or text file)")
parser.add_argument("-f", "--format", dest="format", required=True, help="The evaluation file format (fmmap|mmap|nbest|text)")
parser.add_argument("-v", "--vocab-file", dest="vocab_path", help="The vocabulary file that was used in training")
parser.add_argument("-m", "--model-file", dest="model_path", required=True, help="Input PrimeLM model file")
parser.add_argument("-ulp", "--unnormalized-log-prob-file", dest="ulp_path", help="Output file for sentence-level UNNORMALIZED log-probabilities")
parser.add_argument("-nlp", "--normalized-log-prob-file", dest="nlp_path", help="Output file for sentence-level NORMALIZED log-probabilities")
parser.add_argument("-ppl", "--perplexity", action='store_true', help="Compute perplexity")
parser.add_argument("-op", "--output_path", dest="out_path",  help="Output classes path")
parser.add_argument("-un", "--unnormalized", action='store_true', help="Output need not be normalized")
parser.add_argument("-d", "--device", dest="device", default="gpu", help="The computing device (cpu or gpu)")
parser.add_argument("-lf","--loss-function", dest="loss_function", default="nll", help="Loss function (nll|nce|sll). Default: nll (Negative Log Likelihood)")

args = parser.parse_args()

is_sll = args.loss_function == 'sll'

U.set_theano_device(args.device, 1)

from dlm.models.mlp import MLP
from dlm import eval
from dlm import eval_sll
import theano
import theano.tensor as T

#########################
## Loading model
#

classifier = MLP(model_path=args.model_path)

#########################
## Loading dataset
#

U.xassert(args.format == "mmap" or args.format == "nbest" or args.format == "text" or args.format == "fmmap", "Invalid file format given: " + args.format)
U.xassert(args.perplexity or args.nlp_path or args.ulp_path, "You should use one of (or more) -ppl, -nlp or -ulp")

if args.format == "mmap":
	U.xassert((args.nlp_path is None) and (args.ulp_path is None), "Cannot compute log-probabilities for an mmap file")
	from dlm.io.mmapReader import MemMapReader
	testset = MemMapReader(dataset_path=args.test_path, batch_size=500)
elif args.format == "fmmap":
	U.xassert((args.nlp_path is None) and (args.ulp_path is None), "Cannot compute log-probabilities for an features mmap file")
	from dlm.io.featuresmmapReader import FeaturesMemMapReader
	testset = FeaturesMemMapReader(dataset_path=args.test_path, is_sll=is_sll, batch_size=500)
else:
	U.xassert(args.vocab_path, "Vocab file is required for non-mmap file formats")
	from dlm.io.textReader import TextReader
	is_nbest = False
	if args.format == "nbest":
		is_nbest = True
	testset = TextReader(dataset_path=args.test_path, is_nbest=is_nbest, ngram_size=classifier.ngram_size, vocab_path=args.vocab_path)

#########################
## Compiling theano function
#
if is_sll:
	evaluator = eval_sll.Evaluator(testset, classifier, is_sll=True)
else:
	evaluator = eval.Evaluator(testset, classifier, is_sll=False)
	
#########################
## Testing
#

start_time = time.time()

#########################
'''
print args.loss_function
import theano.tensor as T
import numpy as np
import theano

indices = [0,1,2,4,5]

for i in indices:

	x = testset.get_x(i,args.loss_function)
	testfuncx = theano.function(inputs=[],outputs=[x])
	y = testset.get_y(i,args.loss_function)
	testfuncy = theano.function(inputs=[],outputs=[y])

	# print "############# data x ###############"
	# print testfuncx()
	print "############# data y ###############"
	print testfuncy()

	transitions, scores, delta_0, transitions_tranc, scores_roll, temp_matrix, temp_max_matrix, delta_full_stack, delta, graph, path_end, path_end_index, path, path_before = evaluator.get_batch_predicted_class(i)
	# print "transitions"
	# print transitions
	# print transitions.shape
	# print "transitions_tranc"
	# print transitions_tranc
	# print transitions_tranc.shape
	# print "scores"
	# print scores
	# print scores.shape
	# print "scores_roll"
	# print scores_roll
	# print scores_roll.shape
	# print "temp_matrix"
	# print temp_matrix
	# print temp_matrix.shape
	# print "temp_max_matrix"
	# print temp_max_matrix
	# print temp_max_matrix.shape
	# print "delta_0"
	# print delta_0
	# print delta_0.shape
	# print "delta_full_stack"
	# print delta_full_stack
	# print delta_full_stack.shape
	# print "delta"
	# print delta
	# print delta.shape
	# print "graph"
	# print graph
	# print graph.shape
	print "path_end"
	print path_end
	print path_end.shape
	print "path_end_index"
	print path_end_index
	print path_end_index.shape
	print "path"
	print path
	print path.shape
	print "path_before"
	print path_before
	print path_before.shape

assert False
'''
#########################

if args.perplexity and is_sll:
	L.info("Perplexity: %f" % (evaluator.perplexity()))
	if args.unnormalized:
		L.info("Unnormalized Perplexity: %f" % (evaluator.unnormalized_perplexity()))

if args.nlp_path:
	with open(args.nlp_path, 'w') as output:
		for i in xrange(testset.get_num_sentences()):
			output.write(str(evaluator.get_sequence_log_prob(i)) + '\n')


if args.ulp_path:
	with open(args.ulp_path, 'w') as output:
		for i in xrange(testset.get_num_sentences()):
			output.write(str(evaluator.get_unnormalized_sequence_log_prob(i)) + '\n')

if args.out_path:
	with open(args.out_path, 'w') as output:
		for i in xrange(testset.get_num_batches()):
			batch_labels = evaluator.get_batch_predicted_class(i)
			for label in batch_labels:
				output.write(str(label) + '\n')

L.info("Ran for %.2fs" % (time.time() - start_time))







