import theano.tensor as T
import theano
import dlm.io.logging as L

class SGD:
	def __init__(self, classifier, criterion, learning_rate, trainset, is_sll=False, clip_threshold=0):
		self.eta = learning_rate
		self.is_weighted = trainset.is_weighted
		
		if clip_threshold > 0:
			gparams = [T.clip(T.grad(criterion.cost, param), -clip_threshold, clip_threshold) for param in classifier.params]
		else:
			gparams = [T.grad(criterion.cost, param) for param in classifier.params]
		
		lr = T.fscalar()
		
		updates = [
			(param, param - lr * gparam)
			for param, gparam in zip(classifier.params, gparams)
		]
	
		index = T.lscalar()		# index to a [mini]batch
		x = classifier.input
		y = criterion.y
		
		if self.is_weighted: 
			w = criterion.w
			self.step_func = theano.function(
				inputs=[index, lr],
				outputs=[criterion.cost] + gparams,
				updates=updates,
				givens={
					x: trainset.get_x(index,is_sll),
					y: trainset.get_y(index,is_sll),
					w: trainset.get_w(index,is_sll)
				}
			)
		else:
			self.step_func = theano.function(
				inputs=[index, lr],
				outputs=[criterion.cost, criterion.scores, criterion.delta_0, criterion.delta, criterion.delta_logadd, criterion.transitions_tranc, criterion.delta_matrix, criterion.t] + gparams,
				updates=updates,
				givens={
					x: trainset.get_x(index,is_sll),
					y: trainset.get_y(index,is_sll)
				}
			)

		# self.step_func_test = theano.function(
		# 	inputs=[index],
		# 	outputs=[criterion.delta_logadd],
		# 	givens={
		# 		x: trainset.get_x(index,is_sll),
		# 		y: trainset.get_y(index,is_sll)
		# 	},
		# 	on_unused_input='warn'
		# )

	def step(self, minibatch_index):
		outputs = self.step_func(minibatch_index, self.eta) # original
		# outputs = self.step_func_test(minibatch_index)
		# a = self.step_func_test(minibatch_index)
		# step_cost, gparams = outputs[0], outputs[1:] # original
		a,b,c,d,e,f,g,h, gparams = outputs[0],outputs[1],outputs[2],outputs[3],outputs[4],outputs[5],outputs[6],outputs[7],outputs[7:] # original
		# return a
		# return step_cost, gparams
		return a,b,c,d,e,f,g,h, gparams

	def set_learning_rate(self, eta):
		self.eta = eta
	
	def get_learning_rate(self):
		return self.eta
