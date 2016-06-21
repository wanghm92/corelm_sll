import theano.tensor as T
import theano 

class SentenceLevelLogLikelihood():

	def __init__(self, classifier, args):
		
		self.y = T.ivector('y')

		self.transitions = classifier.A
		
		self.cost = (
			self.calculate_cost_sll(classifier.score_output(), args)
			+ args.L1_reg * classifier.L1
			+ args.L2_reg * classifier.L2_sqr
		)
		
		if args.alpha is not None and args.alpha > 0:
			self.cost = self.cost + args.alpha  * classifier.log_Z_sqr

		self.test = (
			T.mean(classifier.p_y_given_x(self.y))
		)

	def calculate_cost_sll(self, scores, args):
		"""
		Calculates the output and transition deltas for each token, using Sentence Level Likelihood.
		The aim is to minimize the cost:
		C(theta,A) = logadd(scores for all possible paths) - score(correct path)

		:returns: if True, normal gradient calculation was performed.
		    If False, the error was too low and weight correction should be
		    skipped.
		"""

		# ftheta_i,t = network output for i-th tag, at t-th word
		# s = Sum_i(A_tags[i-1],tags[i] + ftheta_i,i), i < len(sentence)   (12)

		self.y_roll = T.cast(T.roll(self.y,1), theano.config.floatX)

		self.y_w = T.cast(T.gt(T.arange(self.y.shape[0]), 0.5), theano.config.floatX)

		self.y_shift = T.cast(self.y_roll * self.y_w, 'int32')

		self.scores = scores # for test

		self.ym1 = self.y-1 # for test

		self.s = scores[T.arange(self.y.shape[0]), self.y-1]

		self.t = self.transitions[self.y_shift, self.y-1]

		self.correct_path_score = T.sum(self.t + self.s)

		# delta[t] = delta_t in equation (14)
		self.delta = self.calculate_delta(scores)

		# logadd_i(delta_T(i)) = log(Sum_i(exp(delta_T(i))))
		# Sentence-level Log-Likelihood (SLL)
		# C(ftheta,A) = logadd_j(s(x, j, theta, A)) - score(correct path)
		self.delta_logadd = T.log(T.sum(T.exp(self.delta), axis=0))
		self.cost = self.delta_logadd - self.correct_path_score

		return self.cost

	def calculate_delta(self, scores):
		"""
		Calculates a matrix with the scores for all possible paths at all given
		points (tokens).
		In the returned matrix, delta[i][j] means the sum of all scores 
		ending in token i with tag j (delta_i(j) in eq. 14 in the paper)
		"""
		# logadd for first token. the transition score of the starting tag must be used.
		# it turns out that logadd = log(exp(score)) = score
		# (use long double because taking exp's leads to very very big numbers)
		# scores[t][k] = ftheta_k,t
		# delta = scores

		# transitions[-1] represents initial transition, A_0,i in paper (mispelled as A_i,0)
		# delta_0(k) = ftheta_k,0 + A_0,i
		# delta[0] += self.transitions[-1]
		self.delta_0 = self.transitions[0] + scores[0] 

		# logadd for the remaining tokens
		# delta_t(k) = ftheta_k,t + logadd_i(delta_t-1(i) + A_i,k)
		#            = ftheta_k,t + log(Sum_i(exp(delta_t-1(i) + A_i,k)))
		self.transitions_tranc = self.transitions[1:].T # A_k,i

		def calculate_delta_recursively(net_scores, delta_prev):
			temp = delta_prev + self.transitions_tranc
			logadd = T.log(T.sum(T.exp(temp), axis=1))
			delta = net_scores + logadd

			return delta

		delta, updates = theano.scan(
			calculate_delta_recursively,
			sequences=[scores],
			outputs_info=[dict(initial=self.delta_0)])
		
		self.delta_matrix = delta

		return delta[-1]


