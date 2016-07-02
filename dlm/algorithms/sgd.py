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
			'''
			self.step_func = theano.function(
				inputs=[index, lr],
				outputs=[criterion.cost] + gparams,
				updates=updates,
				givens={
					x: trainset.get_x(index,is_sll),
					y: trainset.get_y(index,is_sll)
				}
			)
			'''
			self.step_func = theano.function(
				inputs=[index, lr],
				outputs=[criterion.cost, criterion.scores, criterion.scores_roll, criterion.transitions, criterion.delta_0, criterion.temp_matrix, criterion.temp_max_matrix, criterion.logadd_matrix, criterion.delta_matrix, criterion.delta_full_stack, criterion.delta, criterion.delta_logadd, criterion.y_roll, criterion.y_w, criterion.y_shift, criterion.yplus1, criterion.s, criterion.t, criterion.correct_path_score,classifier.looktbout,classifier.hidout, classifier.actout, classifier.hidparams[0],classifier.hidparams[1], classifier.linearparams[0], classifier.linearparams[1]] + gparams,
				updates=updates,
				givens={
					x: trainset.get_x(index,is_sll),
					y: trainset.get_y(index,is_sll)
				}
			)
			
	def step(self, minibatch_index):
		outputs = self.step_func(minibatch_index, self.eta) # original
		# step_cost, gparams = outputs[0], outputs[1:] # original
		a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z, gparams = outputs[0],outputs[1],outputs[2],outputs[3],outputs[4],outputs[5],outputs[6],outputs[7],outputs[8],outputs[9],outputs[10],outputs[11],outputs[12],outputs[13],outputs[14],outputs[15],outputs[16],outputs[17],outputs[18],outputs[19],outputs[20],outputs[21],outputs[22],outputs[23],outputs[24],outputs[25],outputs[26:]
		# return step_cost, gparams # original
		return a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z, gparams

	def set_learning_rate(self, eta):
		self.eta = eta
	
	def get_learning_rate(self):
		return self.eta
