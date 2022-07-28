import pycuda.compiler as comp
import pycuda.driver as cuda
import numpy
import pycuda.autoinit
import time
from tensorflow import keras

#pip install pycuda
#pip install tensorflow

class Net():
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.weights = [[[]]]
        
    def forward(self, input):
        n = len(w0[0]) # number of columns in A / number of rows in B
        n_NP = numpy.int32(n)

        nrA = len(w0) # number of rows in A
        ncB = 1 # number of cols in B

        #matrixMul(d_gpu,a_gpu,b_gpu,block=(4,4,1))
        multiply_them(n1_gpu, w0_gpu, img_gpu, n_NP, block=(ncB,nrA,1))
        sigmoid(n1_gpu,block=(4,1,1))

        n = len(w1[0])
        n_NP = numpy.int32(n)

        nrA = len(w1) # number of rows in A
        ncB = 1 # number of cols in B

        multiply_them(n2_gpu, w1_gpu, n1_gpu, n_NP, block=(ncB,nrA,1))
        sigmoid(n2_gpu,block=(len(n2),1,1))

        cuda.memcpy_dtoh(n2,n2_gpu)
        cuda.memcpy_dtoh(n1,n1_gpu)
        return 0

print("ffs")

mod = comp.SourceModule(
    """
__global__ void multiply_them(float *d, float *a, float *b, int n)
{
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  float t = 0;
  for(int i = 0; i < n; i++){
    t += a[(row * n) + i] * b[col + (blockDim.x * i)];
  }
  d[(row * blockDim.x) + col] = t;
}

__global__ void add_them(float *d, float *a)
{
  int i = threadIdx.x;
  d[i] = a[i] + d[i];
}

__global__ void array_mulitply(float *d, float *a, float *b)
{
  const int i = threadIdx.x;
  d[i] = -a[i] * b[i];
}

__global__ void minus_them(float *d, float *a)
{
  const int i = threadIdx.x;
  atomicAdd(&d[0], -a[i]);
}

__global__ void relu(float *a)
{
  const int i = threadIdx.x;
  if(a[i] < 0) 
  {
    a[i] = 0;
  }
}

__global__ void matrixMul(int *d, int *a, int *b)
{
  const int i = threadIdx.x;
  const int j = threadIdx.y;
  d[i] = threadIdx.x;
}

__global__ void sigmoid(float *d)
{
  const int i = threadIdx.x;
  d[i] = 1 / (1 + exp(-d[i]));
}

__global__ void der_sigmoid(float *d, float *a)
{
  const int i = threadIdx.x;
  d[i] = a[i] * (1 - a[i]);
}
"""
)

MAX_THREADS_PER_BLOCK = \
    cuda.Device(0).get_attribute(pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK)

multiply_them = mod.get_function("multiply_them")
add_them = mod.get_function("add_them")
minus_them = mod.get_function("minus_them")
matrixMul = mod.get_function("matrixMul")
relu = mod.get_function("relu")
sigmoid = mod.get_function("sigmoid")
der_sigmoid = mod.get_function("der_sigmoid")
array_mulitply = mod.get_function("array_mulitply")

# --- testing cuda matrix multiplication ---
input = numpy.random.rand(784,1).astype(numpy.float32)
input_gpu = cuda.mem_alloc(input.nbytes)
cuda.memcpy_htod(input_gpu, input)

weights = numpy.random.rand(4,784).astype(numpy.float32)
weights_gpu = cuda.mem_alloc(weights.nbytes)
cuda.memcpy_htod(weights_gpu, weights)

nodes = numpy.empty((4,1),dtype=numpy.float32); nodes.fill(0)
nodes_gpu = cuda.mem_alloc(nodes.nbytes)
cuda.memcpy_htod(nodes_gpu, nodes)

nodes = numpy.matmul(weights,input)
multiply_them(nodes_gpu,weights_gpu,input_gpu, numpy.int32(784), block=(1,4,1))
print("nodes = ",nodes)

cuda.memcpy_dtoh(nodes, nodes_gpu)
print("nodes gpu = ",nodes)

#---- mnist stuff ---- 

(img_train, label_train), (img_test, label_test) = keras.datasets.mnist.load_data()

img_train = img_train / 255
img_test = img_test / 255

w0=numpy.empty((4,784)).astype(numpy.float32); w0.fill(1)
w1=numpy.empty((10,4)).astype(numpy.float32); w1.fill(1)

#weightsFile = "relu-untrained-weights784-4-10.txt"
#weightsFile = "relu-weights784-4-10.txt"
weightsFile = "sigmoid-untrained-weights784-4-10.txt"
#weightsFile = "sigmoid-weights784-4-10.txt"

f = open(weightsFile, "r")
lines = f.readlines()[1:785]
i = 0
for line in lines:
  line = line.replace("\n","")
  array = line.split(",")
  for j in range(len(array)):
    w0[j][i] = array[j]
  i+=1

f = open(weightsFile, "r")
lines = f.readlines()[785:]
i = 0
for line in lines:
  line = line.replace("\n","")
  array = line.split(",")
  for j in range(len(array)):
    w1[j][i] = array[j]
  i+=1

testNet = Net()
testNet.weights[0] = w0
testNet.weights.append(w1)

w0_gpu = cuda.mem_alloc(w0.nbytes)
cuda.memcpy_htod(w0_gpu, w0)
w1_gpu = cuda.mem_alloc(w1.nbytes)
cuda.memcpy_htod(w1_gpu, w1)

n1 = numpy.zeros((4, 1),dtype=numpy.float32)
n1_gpu = cuda.mem_alloc(n1.nbytes)
cuda.memcpy_htod(n1_gpu,n1)

n1input = numpy.zeros((4, 1),dtype=numpy.float32)
n1input_gpu = cuda.mem_alloc(n1input.nbytes)
cuda.memcpy_htod(n1input_gpu,n1input)

n2 = numpy.zeros((10, 1),dtype=numpy.float32)
n2_gpu = cuda.mem_alloc(n2.nbytes)
cuda.memcpy_htod(n2_gpu,n2)

n2input = numpy.zeros((10, 1),dtype=numpy.float32)
n2input_gpu = cuda.mem_alloc(n2input.nbytes)
cuda.memcpy_htod(n2input_gpu,n2input)
# --- training ---

w0grads = numpy.zeros_like(w0)
w1grads = numpy.zeros_like(w1)

outputLoss = numpy.zeros((10),dtype=numpy.float32)
learningRate = 0.1
for epoch in range(1):

  correct = 0
  start_time = time.time()
  for i in range(len(img_train)):

    trainImg = img_train[i]
    trainImg32 = trainImg.astype(numpy.float32)

    n0 = trainImg32.flatten()

    img_gpu = cuda.mem_alloc(trainImg32.nbytes)
    cuda.memcpy_htod(img_gpu,trainImg32)

    #last weights

    testNet.forward(img_gpu)
    der_sigmoid(n1input_gpu,n1_gpu,block=(4,1,1))
    der_sigmoid(n2input_gpu,n2_gpu,block=(10,1,1))
    cuda.memcpy_dtoh(n1input, n1input_gpu)
    cuda.memcpy_dtoh(n2input, n2input_gpu)

    guess = n2.argmax()
    if guess == label_train[i]:
      correct+=1

    for j in range(10):
      if j == label_train[i]:
        outputLoss[j] = 1 - n2[j]
      else:
        outputLoss[j] = 0 - n2[j]

    outputLoss_gpu = cuda.mem_alloc(outputLoss.nbytes)
    cuda.memcpy_htod(outputLoss_gpu,outputLoss)

    outputLossInput = numpy.zeros_like(outputLoss)
    outputLossInput_gpu = cuda.mem_alloc(outputLossInput.nbytes)
    cuda.memcpy_htod(outputLossInput_gpu,outputLossInput)

    array_mulitply(outputLossInput_gpu,outputLoss_gpu,n2input_gpu,block=(10,1,1))

    w1grads_gpu = cuda.mem_alloc(w1grads.nbytes)
    cuda.memcpy_htod(w1grads_gpu,w1grads)

    n = 1
    n_NP = numpy.int32(n)
    multiply_them(w1grads_gpu, outputLossInput_gpu, n1_gpu, n_NP, block=(4,10,1))
    cuda.memcpy_dtoh(w1grads,w1grads_gpu)

    add_them(w1_gpu, w1grads_gpu,block=(40,1,1))
    cuda.memcpy_dtoh(w1,w1_gpu)


  
    #backward first weights
    '''
    for y in range(len(w0[0])):
      for x in range(len(w0)):
        totalError = 0
        input = n1input[x]
        for n in range(len(n2)):
          totalError += (w1[n][x] * w1grads[n][x]) / len(n2)
        totalError = totalError * input * n0[y]
        w0grads[x][y] += totalError

    #optimize first weights
    for y in range(len(w0grads[0])):
      for x in range(len(w0grads)):
        w0[x][y] -= w0grads[x][y] * learningRate

    '''
    w1grads = numpy.zeros_like(w1)    
    #w0grads = numpy.zeros_like(w0)
    

#    for x in range(len(w1)):
#      for y in range(len(w1[x])):
#        prevOutput = n1[y]
#        output = n2[y]
#        input = n2input[y]
#        w1grads[x][y] = -outputLoss[y] * input * prevOutput
#        w1[x][y] -= w1grads[x][y] * learningRate

    cuda.memcpy_htod(w0_gpu, w0)
    cuda.memcpy_htod(w1_gpu, w1)

  print("--- %s seconds ---" % (time.time() - start_time))
  print("correct = ",(correct/len(img_train)))

# --- testing ---

correct = 0
start_time = time.time()
for i in range(len(img_test)):
  testImg = img_test[i]

  testImg32 = testImg.astype(numpy.float32)
  
  img_gpu = cuda.mem_alloc(testImg32.nbytes)
  cuda.memcpy_htod(img_gpu, testImg32)

  testNet.forward(img_gpu)

  guess = n2.argmax()
  if guess == label_test[i]:
    correct +=1
  #guess = output.index(max(output))
  #print("guess = ",guess)
print("--- %s seconds ---" % (time.time() - start_time))
print("test dataset: correct = ",(correct/len(img_test)))

# --------

#a=numpy.empty(1024, dtype=numpy.float32); a.fill(numpy.float32(1))
a=numpy.matrix('4 8 9 1 2; 2 8 9 1 2; 8 7 5 7 1', dtype=numpy.float32)
b=numpy.matrix('8; 7; 4; 2; 5', dtype=numpy.float32)
d=numpy.matrix('0; 0; 0', dtype=numpy.float32)

a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

b_gpu = cuda.mem_alloc(b.nbytes)
cuda.memcpy_htod(b_gpu, b)

d_gpu = cuda.mem_alloc(d.nbytes)
cuda.memcpy_htod(d_gpu, d)

n = 5
n_NP = numpy.int32(n)

nrA = 3 # number of rows in A
ncB = 1 # number of cols in B

#matrixMul(d_gpu,a_gpu,b_gpu,block=(4,4,1))
multiply_them(d_gpu, a_gpu, b_gpu, n_NP, block=(ncB,nrA,1))

cuda.memcpy_dtoh(d, d_gpu)
for i in range(len(d)):
  print(i," : ", d[i])

