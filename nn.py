import random
from math import exp 
random.seed(1234) # TODO: Figure out how much the seed impacts the
# generated network and adjust this accordingly.
# TODO: Move seed() to appropriate location as per above ^^
print(random.random())
class NN:
	class __layer:
		def sigmoid(self, x):
			return 1.0 / (1.0 + exp(-x))

		def __init__(self, dim_in, dim_out, isSigmoid,
				alpha = 0.3, sigma = 0.6, seed = random.random):
			self.dim_in = dim_in
			self.dim_out = dim_out
			self.weights = [seed() for i in range((1 + dim_in) * dim_out)]
			self.dweights = [0 for i in range(len(self.weights))]
			self.isSigmoid = isSigmoid
			self.alpha = alpha
			self.sigma = sigma
			print(self.weights)

		def run(self, inp):
			assert len(inp) == self.dim_in
			self.inp = inp[:]
			self.inp.append(1)
			output = []
			#print(self.weights)
			#print(inp)
			for i in range(self.dim_out):
				output.append(0)
				for j in range(self.dim_in + 1):
					output[i] += self.inp[j] * self.weights[i*self.dim_in+j]
				if self.isSigmoid:
					#print(output[i], "o")
					output[i] = self.sigmoid(output[i])
			#print (self.weights)
			self.output = output[:]
			return output

		def train(self, error):
			#print(len(error), self.dim_out, "K", error)
			#assert len(error) == self.dim_out 
			nextError = []
			for i in range(self.dim_out):
				d = error[i]
				if (self.isSigmoid):
					d *= self.output[i] * (1 - self.output[i])
				for j in range(self.dim_in + 1):
					nextError.append(0)
					idx = i * self.dim_in + j
					nextError[j] += self.weights[idx] * d
					dw = self.inp[j] * d * self.alpha
					self.weights[idx] += self.dweights[idx]*self.sigma + dw
					self.dweights[idx] = dw
			return nextError




	def __init__(self, dims, seed = random.random, alpha = 0.3, sigma = 0.6):
		self.dims = dims;
		assert len(dims) > 2, "A Neural Network needs two or more layers"
		self.layers = [self.__layer(dims[i], dims[i + 1], False, seed = seed, alpha = alpha, sigma = sigma)
			for i in range(len(dims) - 1)
		]
	
	def run(self, inp):
		for layer in self.layers:
			output = layer.run(inp)
			inp = output
		return output
	
	def train(self, inp, target):
		output = self.run(inp)
		error = []
		for i in range(len(output)):
			error.append((target[i] - output[i]))# * abs(target[i] - output[i]))
		for layer in reversed(self.layers):
			#print(error)
			error = layer.train(error)

def rseed():
	return (random.random() - 0.5) * 4

def test():
	nn = NN([2, 2, 1], seed = rseed, sigma = 0.6)#, seed=lambda: 0)
	nn.layers[0].isSigmoid = True
	tr = [[[0, 1], [1]], [[1, 0], [1]], [[1, 1], [0]], [[0, 0], [0]]]
	for i in range(2001):
		c = random.choice(tr)
		nn.train(c[0], c[1])
		#print(c[0], nn.run(c[0]), "res")
		if i % 100 == 0:
			print ("Generation {0}".format(int(i / 100)))
			for track in tr:
				print("Input {0}, output {1}, target {2}".format(track[0], nn.run(track[0]), track[1]))
	print("I ran")

if __name__ == "__main__":
		test()
