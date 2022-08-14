import pycuda.compiler as comp
import pycuda.driver as cuda
import numpy
import pycuda.autoinit
import time
from tensorflow import keras
from os.path import exists

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

        multiply_them(n1_gpu, w0_gpu, n0_gpu, n_NP, block=(ncB,nrA,1))
        sigmoid(n1_gpu,block=(len(n1),1,1))

        n = len(w1[0])
        n_NP = numpy.int32(n)

        nrA = len(w1) # number of rows in A
        ncB = 1 # number of cols in B

        multiply_them(n2_gpu, w1_gpu, n1_gpu, n_NP, block=(ncB,nrA,1))
        sigmoid(n2_gpu,block=(len(n2),1,1))
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

__global__ void optimize(float *d, float *a, float lr)
{
  int i = threadIdx.x + (blockDim.x * blockIdx.x);
  d[i] = (lr * -a[i]) + d[i];
}

__global__ void array_mulitply_minus(float *d, float *a, float *b)
{
  const int i = threadIdx.x;
  d[i] = -a[i] * b[i];
}

__global__ void array_mulitply(float *d, float *a, float *b)
{
  const int i = threadIdx.x + (blockDim.x * blockIdx.x);
  d[i] = a[i] * b[i];
}


__global__ void get_output_loss(float *d, float *o, int a)
{
  int i = threadIdx.x;
  if(i == a) {
    d[i] = 1 - o[i];
  } else {
    d[i] = 0 - o[i];
  }
}

__global__ void get_node_loss(float *d, float *a, int n, int s)
{
  int i = threadIdx.x;
  float t = 0;
  for(int j = 0; j < n; j++) 
  {
    t += a[i + j*blockDim.x];
  } 
  d[i] = t / s;
}

__global__ void get_grads(float *d, float *a, float *b, float *c)
{
  d[threadIdx.x + blockDim.x * blockIdx.x] = a[blockIdx.x] * b[blockIdx.x] * c[threadIdx.x]; 
}

__global__ void check_answer(int *a, float *output, int answer)
{
  for(int i = 0; i < 10; i++)
  {
    if(output[i] > output[answer])
    {
      return;
    }
  }
  a[0] = a[0] + 1;
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
optimize = mod.get_function("optimize")
minus_them = mod.get_function("minus_them")
relu = mod.get_function("relu")
sigmoid = mod.get_function("sigmoid")
der_sigmoid = mod.get_function("der_sigmoid")
array_mulitply_minus = mod.get_function("array_mulitply_minus")
array_mulitply = mod.get_function("array_mulitply")
get_output_loss = mod.get_function("get_output_loss")
get_node_loss = mod.get_function("get_node_loss")
get_grads = mod.get_function("get_grads")
check_answer = mod.get_function("check_answer")

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

n0size = 784
n1size = 16
n2size = 10

img_train = img_train / 255
img_test = img_test / 255

w0=numpy.empty((n1size,n0size)).astype(numpy.float32); w0.fill(1)
w1=numpy.empty((n2size,n1size)).astype(numpy.float32); w1.fill(1)

layers = [784,16,10]

#weightsFile = "relu-untrained-weights784-4-10.txt"
#weightsFile = "relu-weights784-4-10.txt"
#weightsFile = "sigmoid-untrained-weights784-4-10.txt"
#weightsFile = "sigmoid-weights784-4-10.txt"
#weightsFile = "sigmoid-untrained-weights784-16-10.txt"
weightsFile = "sigmoid-untrained-weights"
#weightsFile = "sigmoid-weights784-16-10.txt"

for i in range(len(layers) - 1):
  weightsFile += str(layers[i]) + "-"
weightsFile += str(layers[len(layers)-1]) + ".txt"

if exists(weightsFile):
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

else:
  for a in range(len(w1)):
    for b in range(len(w1[0])):
      w1[a][b] = numpy.random.uniform() * (2 / numpy.sqrt(n1size)) - 1 / numpy.sqrt(n1size)
  for a in range(len(w0)):
    for b in range(len(w0[0])):
      w0[a][b] = numpy.random.uniform() * (2 / numpy.sqrt(n0size)) - 1 / numpy.sqrt(n0size)


testNet = Net()
testNet.weights[0] = w0
testNet.weights.append(w1)

w0_gpu = cuda.mem_alloc(w0.nbytes)
cuda.memcpy_htod(w0_gpu, w0)
w1_gpu = cuda.mem_alloc(w1.nbytes)
cuda.memcpy_htod(w1_gpu, w1)

n1 = numpy.zeros((n1size, 1),dtype=numpy.float32)
n1_gpu = cuda.mem_alloc(n1.nbytes)
cuda.memcpy_htod(n1_gpu,n1)

n1input = numpy.zeros((n1size, 1),dtype=numpy.float32)
n1input_gpu = cuda.mem_alloc(n1input.nbytes)
cuda.memcpy_htod(n1input_gpu,n1input)

n2 = numpy.zeros((n2size, 1),dtype=numpy.float32)
n2_gpu = cuda.mem_alloc(n2.nbytes)
cuda.memcpy_htod(n2_gpu,n2)

n2input = numpy.zeros((n2size, 1),dtype=numpy.float32)
n2input_gpu = cuda.mem_alloc(n2input.nbytes)
cuda.memcpy_htod(n2input_gpu,n2input)
# --- training ---

w0grads = numpy.zeros_like(w0)
w1grads = numpy.zeros_like(w1)
w1Loss = numpy.zeros_like(w1)

w1grads_gpu = cuda.mem_alloc(w1grads.nbytes)
cuda.memcpy_htod(w1grads_gpu,w1grads)

w1Loss_gpu = cuda.mem_alloc(w1Loss.nbytes)
cuda.memcpy_htod(w1Loss_gpu,w1Loss)

w0grads_gpu = cuda.mem_alloc(w0grads.nbytes)
cuda.memcpy_htod(w0grads_gpu,w0grads)

outputLoss = numpy.zeros((len(n2)),dtype=numpy.float32)
outputLoss_gpu = cuda.mem_alloc(outputLoss.nbytes)
outputLossInput = numpy.zeros_like(outputLoss) #outputLoss * input
outputLossInput_gpu = cuda.mem_alloc(outputLossInput.nbytes)
cuda.memcpy_htod(outputLossInput_gpu,outputLossInput)

totalErrors = numpy.zeros((len(n1)),dtype=numpy.float32)
totalErrors_gpu = cuda.mem_alloc(totalErrors.nbytes)
cuda.memcpy_htod(totalErrors_gpu,totalErrors)

n0 = numpy.zeros((n0size),dtype=numpy.float32)
n0_gpu = cuda.mem_alloc(n0.nbytes)
cuda.memcpy_htod(n0_gpu,n0)

training_correct = numpy.zeros((1),dtype=numpy.int32)
training_correct_gpu = cuda.mem_alloc(training_correct.nbytes)
cuda.memcpy_htod(training_correct_gpu,training_correct)

test_correct = numpy.zeros((1),dtype=numpy.int32)
test_correct_gpu = cuda.mem_alloc(test_correct.nbytes)
cuda.memcpy_htod(test_correct_gpu,test_correct)

totalErrors = numpy.zeros((len(n1)),dtype=numpy.float32)

learningRate = numpy.float32(0.1)
for epoch in range(1):

  correct = 0
  start_time = time.time()
  for i in range(len(img_train)):
    trainImg = img_train[i]
    trainImg32 = trainImg.astype(numpy.float32)
    cuda.memcpy_htod(n0_gpu,trainImg32)

    #last weights

    testNet.forward(n0_gpu)
    der_sigmoid(n1input_gpu,n1_gpu,block=(len(n1),1,1))
    der_sigmoid(n2input_gpu,n2_gpu,block=(len(n2),1,1))

    check_answer(training_correct_gpu, n2_gpu, numpy.int32(label_train[i]),block=(1,1,1))

    #backward

    get_output_loss(outputLoss_gpu, n2_gpu, numpy.int32(label_train[i]), block=(len(n2),1,1))
    
    array_mulitply_minus(outputLossInput_gpu,outputLoss_gpu,n2input_gpu,block=(len(outputLoss),1,1))

    n = 1
    n_NP = numpy.int32(n)
    multiply_them(w1grads_gpu, outputLossInput_gpu, n1_gpu, n_NP, block=(len(n1),len(outputLoss),1))

    #backward first weights ???
    
    array_mulitply(w1Loss_gpu,w1_gpu,w1grads_gpu,block=((len(n1) * len(n2)),1,1))

    get_node_loss(totalErrors_gpu,w1Loss_gpu,numpy.int32(len(n2)),numpy.int32(len(n2)),block=(len(n1),1,1))

    get_grads(w0grads_gpu,totalErrors_gpu,n1input_gpu,n0_gpu,block=(len(n0),1,1),grid=(len(n1),1))

    #optimize
  
    optimize(w0_gpu,w0grads_gpu,learningRate,block=(len(n0),1,1),grid=(len(n1),1))

    optimize(w1_gpu, w1grads_gpu,learningRate,block=(len(n1),1,1),grid=(len(n2),1))

  print("--- %s seconds ---" % (time.time() - start_time))
  cuda.memcpy_dtoh(training_correct,training_correct_gpu)
  print("correct (GPU) = ",(training_correct[0]/len(img_train)))

# --- testing ---

correct = 0
start_time = time.time()
for i in range(len(img_test)):
  testImg = img_test[i]
  testImg32 = testImg.astype(numpy.float32)
  
  n0_gpu = cuda.mem_alloc(testImg32.nbytes)
  cuda.memcpy_htod(n0_gpu, testImg32)

  testNet.forward(n0_gpu)

  check_answer(test_correct_gpu, n2_gpu, numpy.int32(label_test[i]),block=(1,1,1))
  #guess = output.index(max(output))
  #print("guess = ",guess)
print("--- %s seconds ---" % (time.time() - start_time))
cuda.memcpy_dtoh(test_correct,test_correct_gpu)
print("test dataset: correct = ",(test_correct[0]/len(img_test)))

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

multiply_them(d_gpu, a_gpu, b_gpu, n_NP, block=(ncB,nrA,1))

cuda.memcpy_dtoh(d, d_gpu)
for i in range(len(d)):
  print(i," : ", d[i])