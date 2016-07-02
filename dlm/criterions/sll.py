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

		self.scores = scores

		self.yplus1 = self.y + 1

		self.s = scores[T.arange(self.y.shape[0]), self.y]

		self.y_roll = T.cast(T.roll(self.yplus1,1), theano.config.floatX)

		self.y_w = T.cast(T.gt(T.arange(self.y.shape[0]), 0.5), theano.config.floatX)

		self.y_shift = T.cast(self.y_roll * self.y_w, 'int32')

		self.t = self.transitions[self.y_shift, self.y]

		self.correct_path_score = T.sum(self.t + self.s)

		self.delta = self.calculate_delta(scores)

		self.delta_max = T.max(self.delta)
		self.delta_logadd = T.log(T.sum(T.exp(self.delta - self.delta_max), axis=0)) + self.delta_max

		self.cost = self.delta_logadd - self.correct_path_score #) / (T.cast(self.y.shape[0], theano.config.floatX))

		return self.cost

	def calculate_delta(self, scores):

		self.delta_0 = self.transitions[0] + scores[0]

		# self.transitions_max = T.max(self.transitions[0])
		# self.transitions_logadd = T.log(T.sum(T.exp(self.transitions[0] - self.transitions_max), axis=0)) + self.transitions_max
		# self.delta_0 = self.transitions_logadd + scores[0]
			
		self.transitions_tranc = self.transitions[1:].T # A_k,i
		self.scores_roll = T.roll(scores, -scores.shape[1])

		def calculate_delta_recursively(net_scores, delta_prev):
			temp = delta_prev + self.transitions_tranc
			temp_max = T.max(temp, axis=1)
			logadd = T.log(T.sum(T.exp(temp.T-temp_max), axis=0)) + temp_max
			delta = net_scores + logadd

			return temp, temp_max, logadd, delta

		[temp, temp_max, logadd, delta], updates = theano.scan(
			calculate_delta_recursively,
			sequences=[self.scores_roll],
			outputs_info=[None,None,None, dict(initial=self.delta_0)])
		
		self.temp_matrix = temp
		self.temp_max_matrix = temp_max
		self.logadd_matrix = logadd
		self.delta_matrix = delta

		self.delta_flat = T.flatten(delta)
		self.delta_full_flat = T.concatenate([self.delta_0, self.delta_flat],axis=0)
		self.delta_full_stack = T.reshape(self.delta_full_flat,(delta.shape[0]+1,delta.shape[1]))
		
		return self.delta_full_stack[-2]



