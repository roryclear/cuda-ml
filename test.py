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
        self.weights = None
        self.nodes = None
        self.grads = None
        self.nodesInput = None
        self.loss = None

        self.weights_gpu = None
        self.nodes_gpu = None
        self.grad_gpu = None
        self.nodesInput_gpu = None
        self.loss_gpu = None

    def copyToDevice(self):
      self.weights_gpu = cuda.mem_alloc(self.weights.nbytes)
      self.nodes_gpu = cuda.mem_alloc(self.nodes.nbytes)
      self.grads_gpu = cuda.mem_alloc(self.grads.nbytes)
      self.nodesInput_gpu = cuda.mem_alloc(self.nodesInput.nbytes)
      self.loss_gpu = cuda.mem_alloc(self.loss.nbytes)

      cuda.memcpy_htod(self.nodes_gpu,self.nodes)
      cuda.memcpy_htod(self.weights_gpu,self.weights)
      cuda.memcpy_htod(self.grads_gpu,self.grads)
      cuda.memcpy_htod(self.nodesInput_gpu,self.nodesInput)
      cuda.memcpy_htod(self.loss_gpu,self.loss)


    def optimize(self):
      length = len(weights)
      bx = length
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024
      optimize(self.weights_gpu, self.grads_gpu,learningRate, numpy.int32(length), block=(bx,1,1),grid=(gx,1))

    def zero_grad(self):
      length = len(weights)
      bx = length
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024
      reset_values(self.grads_gpu,numpy.int32(length),block=(bx,1,1),grid=(gx,1))
  
    def backward(self):
      length = len(self.nodesInput)
      bx = length
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024
      der_sigmoid(self.nodesInput_gpu,self.nodes_gpu, numpy.int32(length),block=(bx,1,1),grid=(gx,1))

      start = numpy.int32(self.layers[0] + self.layers[1])
      check_answer(training_correct_gpu, self.nodes_gpu, start, numpy.int32(label_train[i]),block=(1,1,1))

      #backward
      start = numpy.int32(self.layers[0] + self.layers[1])
      get_output_loss(self.loss_gpu, self.nodes_gpu, start, numpy.int32(label_train[i]),
                      block=(self.layers[2],1,1))
      
      n = 1
      n_NP = numpy.int32(n)

      bx = self.layers[1]
      by = self.layers[2]
      gx = 1
      gy = 1

      bxn = bx
      byn = by
      if bx*by > 1024:
        if bx > by:
          bxn = int(1024 / by)
          gx = int(bx / bxn) + 1
          
      #int ncA, int ncB, int nrA
      startn0 = numpy.int32(self.layers[0])
      startD = numpy.int32(self.layers[0] * self.layers[1])
      startW = numpy.int32(self.layers[0] + self.layers[1])
      #multiply_them_index2(float *nodesD, float *weights, float *nodesA, int ncA, int ncB, int nrA, int startn0, int startD, int startW)
      multiply_them_index_add(self.grads_gpu, self.loss_gpu, self.nodesInput_gpu,
       self.nodes_gpu, n_NP, numpy.int32(bx), numpy.int32(10), startn0, startD, startW,
        block=(bxn,by,1), grid=(gx,gy)) 
      
      #backward first weights ???
      length = self.layers[1] * self.layers[2]
      bx = length
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024
      startD = numpy.int32(self.layers[0] * self.layers[1])
      startA = numpy.int32(self.layers[0] * self.layers[1])
      startB = numpy.int32(self.layers[0] * self.layers[1])
      array_mulitply(self.loss_gpu,self.weights_gpu,self.grads_gpu,startD,startA,startB,numpy.int32(length)
      ,block=(bx,1,1),grid=(gx,1))

      startA = numpy.int32(self.layers[0] * self.layers[1])
      length = self.layers[1]  
      bx = length
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024
      get_node_loss(self.loss_gpu,self.loss_gpu,numpy.int32(self.layers[2]),startA,
                    numpy.int32(length),block=(bx,1,1),grid=(gx,1))

      startB = numpy.int32(self.layers[0])
      startC = numpy.int32(0)
      get_grads(self.grads_gpu,self.loss_gpu,self.nodesInput_gpu, self.nodes_gpu,startB,startC,
                block=(self.layers[0],1,1),grid=(self.layers[1],1))

    def forward(self):

      #copy input (n0_gpu) to nodes_gpu
      length = self.layers[0]
      bx = length
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024
      
      copy(self.nodes_gpu, img_gpu, numpy.int32(0), numpy.int32(0), numpy.int32(length), block=(bx,1,1), grid=(gx,1))

      
      n = self.layers[0] # number of columns in A / number of rows in B
      n_NP = numpy.int32(n)
      bx = 1 # number of cols in B
      by = self.layers[1] # number of rows in A
      gx = 1
      gy = 1
      if by > 1024:
        gy = int(by / 1024) + 1
        by = 1024
      startn0 = numpy.int32(0)
      startn1 = numpy.int32(self.layers[0])
      startw = numpy.int32(0)
      multiply_them_index(self.nodes_gpu, self.weights_gpu, self.nodes_gpu, n_NP, numpy.int32(bx) 
      ,numpy.int32(self.layers[1]) , startn0, startn1,
                            startw, block=(bx,by,1), grid=(gx,gy))

      
      bx = self.layers[1]
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024

      sigmoid_index(self.nodes_gpu,numpy.int32(self.layers[0]),numpy.int32(self.layers[1]),
                    block=(bx,1,1), grid=(gx,1))

      n = self.layers[1]
      n_NP = numpy.int32(n)

      bx = 1 # number of cols in B
      by = self.layers[2] # number of rows in A
      gx = 1
      gy = 1

      startn0 = numpy.int32(self.layers[0])
      startn1 = numpy.int32(self.layers[0] + self.layers[1])
      startW = numpy.int32(self.layers[0] * self.layers[1])
      multiply_them_index(self.nodes_gpu, self.weights_gpu, self.nodes_gpu, n_NP, numpy.int32(bx),
                          numpy.int32(self.layers[2]), startn0, startn1, startW, block=(bx,by,1), grid=(gx,gy))

      startA = numpy.int32(self.layers[0] + self.layers[1])
      startD = numpy.int32(0)
      length = self.layers[2]
      bx = length
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024

      start = numpy.int32(self.layers[0] + self.layers[1])
      length = numpy.int32(self.layers[2])
      bx = int(length)
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024
      sigmoid_index(self.nodes_gpu, start, length, block=(bx,1,1), grid=(gx,1))
      return 0

mod = comp.SourceModule(
    """
  __global__ void multiply_them_index(float *nodesD, float *weights, float *nodesA, int ncA, int ncB, int nrA, int startn0, int startD, int startW)
{
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  float t = 0;
  if(col < ncB && row < nrA)
  {
  for(int i = 0; i < ncA; i++){
    t += weights[startW + (row * ncA) + i] * nodesA[startn0 + col + (i * ncB)];
  }
    nodesD[startD + (row * ncB) + col] = t;
  }
}


__global__ void multiply_them_index_add(float *nodesD, float *weights, float *input ,float *nodesA, int ncA, int ncB, int nrA, int startn0, int startD, int startW)
{
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  float t = 0;
  if(col < ncB && row < nrA)
  {
  for(int i = 0; i < ncA; i++){
    t += -weights[startW + (row * ncA) + i] * input[startW + (row * ncA) + i] * nodesA[startn0 + col + (i * ncB)];
  }
    nodesD[startD + (row * ncB) + col] += t;
  }
}

__global__ void multiply_them(float *d, float *a, float *b, int ncA, int ncB, int nrA)
{
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;

  float t = 0;
  if(col < ncB && row < nrA)
  {
  for(int i = 0; i < ncA; i++){
    t += a[(row * ncA) + i] * b[col + (i * ncB)];
  }
  }
  d[(row * ncB) + col] = t;
}

__global__ void multiply_them_add(float *d, float *a, float *b, int ncA, int ncB, int nrA)
{
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;

  float t = 0;
  if(col < ncB && row < nrA)
  {
  for(int i = 0; i < ncA; i++){
    t += a[(row * ncA) + i] * b[col + (i * ncB)];
  }
  }
  d[(row * ncB) + col] += t;
}

__global__ void optimize(float *d, float *a, float lr, int length)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < length)
  {
  d[i] = (lr * -a[i]) + d[i];
  }
}

__global__ void array_mulitply_minus(float *d, float *a, float *b, int startb)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  d[i] = -a[startb + i] * b[startb + i];
}

__global__ void array_mulitply(float *d, float *a, float *b, int startD, int startA, int startB, int length)
{
  const int i = threadIdx.x + (blockDim.x * blockIdx.x);
  if(i < length)
  {
  d[startD + i] = a[startA + i] * b[startB + i];
  }
}


__global__ void get_output_loss(float *d, float *o, int start, int a)
{
  int i = threadIdx.x;
  if(i == a) {
    d[start + i] = 1 - o[start + i];
  } else {
    d[start + i] = 0 - o[start + i];
  }
}

__global__ void get_node_loss(float *d, float *a, int n, int startA, int length)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  float t = 0;
  for(int j = 0; j < n; j++) 
  {
    if(i < length)
    {
    t += a[startA + i + j*length];
    }
  }
  if(i < length)
  { 
  d[i] = t / n;
  }
}

__global__ void get_grads(float *d, float *a, float *b, float *c, int startB, int startC)
{
  d[threadIdx.x + blockDim.x * blockIdx.x] += a[blockIdx.x] * b[startB + blockIdx.x] * c[startC + threadIdx.x]; 
}

__global__ void reset_values(float *d, int length)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < length)
  {
    d[i] = 0;
  }
}

__global__ void reset_values_index(float *d, int start, int length)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < length)
  {
    d[i + start] = 0;
  }
}


__global__ void check_answer(int *a, float *output, int start,int answer)
{
  for(int i = 0; i < 10; i++)
  {
    if(output[start + i] > output[start + answer])
    {
      return;
    }
  }
  a[0] = a[0] + 1;
}

__global__ void add_them(float* d, float *a, int length)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < length)
  {
    d[i] = d[i] + a[i];
  }
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

__global__ void sigmoid(float *d, int length)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < length)
  {
    d[i] = 1 / (1 + exp(-d[i]));
  }
}

__global__ void sigmoid_index(float *d, int start, int length)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < length)
  {
    d[start + i] = 1 / (1 + exp(-d[start + i]));
  }
}

__global__ void der_sigmoid(float *d, float *a, int length)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < length)
  {
    d[i] = a[i] * (1 - a[i]);
  }
}

__global__ void copy(float *d, float *a, int startA, int startD, int length)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < length)
  {
    d[i + startD] = a[i + startA];
  }
}
"""
)

MAX_THREADS_PER_BLOCK = \
    cuda.Device(0).get_attribute(pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK)

multiply_them = mod.get_function("multiply_them")
multiply_them_index = mod.get_function("multiply_them_index")
multiply_them_index_add = mod.get_function("multiply_them_index_add") #adds to result
multiply_them_add = mod.get_function("multiply_them_add")
optimize = mod.get_function("optimize")
minus_them = mod.get_function("minus_them")
relu = mod.get_function("relu")
sigmoid = mod.get_function("sigmoid")
sigmoid_index = mod.get_function("sigmoid_index")
der_sigmoid = mod.get_function("der_sigmoid")
array_mulitply_minus = mod.get_function("array_mulitply_minus")
array_mulitply = mod.get_function("array_mulitply")
get_output_loss = mod.get_function("get_output_loss")
get_node_loss = mod.get_function("get_node_loss")
get_grads = mod.get_function("get_grads")
reset_values = mod.get_function("reset_values")
reset_values_index = mod.get_function("reset_values_index")
check_answer = mod.get_function("check_answer")
add_them = mod.get_function("add_them")
copy = mod.get_function("copy")


#---- mnist stuff ---- se

(img_train, label_train), (img_test, label_test) = keras.datasets.mnist.load_data()

img_train = img_train / 255
img_test = img_test / 255

testNet = Net()
testNet.layers = [784,16,10]

numberOfNodes = 0
for i in range(len(testNet.layers)):
  numberOfNodes += testNet.layers[i]

testNet.nodesInput = numpy.zeros((numberOfNodes, 1),dtype=numpy.float32)

numberOfWeights = 0
for i in range(len(testNet.layers)-1):
  numberOfWeights += testNet.layers[i] * testNet.layers[i+1]

testNet.nodes = numpy.zeros((numberOfNodes, 1),dtype=numpy.float32)
testNet.grads = numpy.zeros((numberOfWeights, 1),dtype=numpy.float32)
testNet.loss = numpy.zeros((numberOfWeights, 1),dtype=numpy.float32)
testNet.nodesInput = numpy.zeros((numberOfNodes, 1),dtype=numpy.float32)

weights = numpy.zeros((numberOfWeights, 1),dtype=numpy.float32)

weightsFile = "sigmoid-untrained-weights"
#weightsFile = "sigmoid-weights"

for i in range(len(testNet.layers) - 1):
  weightsFile += str(testNet.layers[i]) + "-"
weightsFile += str(testNet.layers[len(testNet.layers)-1]) + "-1d.txt"

if exists(weightsFile):
  f = open(weightsFile, "r")
  lines = f.readlines()
  for i in range(len(lines)):
    line = lines[i].replace("\n","")
    weights[i] = line

else:
  print("no weights file was found")    
  for x in range(len(weights)):
    weights[x] = numpy.random.uniform() * (2 / numpy.sqrt(testNet.layers[0])) - 1 / numpy.sqrt(testNet.layers[0])

testNet.weights = weights

testNet.copyToDevice()

# --- training ---
training_correct = numpy.zeros((1),dtype=numpy.int32)
training_correct_gpu = cuda.mem_alloc(training_correct.nbytes)
cuda.memcpy_htod(training_correct_gpu,training_correct)

test_correct = numpy.zeros((1),dtype=numpy.int32)
test_correct_gpu = cuda.mem_alloc(test_correct.nbytes)
cuda.memcpy_htod(test_correct_gpu,test_correct)

trainImg32 = img_train[0].astype(numpy.float32)
img_gpu = cuda.mem_alloc(trainImg32.nbytes)


learningRate = numpy.float32(0.1)
batchSize = 1
for epoch in range(1):

  correct = 0
  start_time = time.time()
  for i in range(len(img_train)): 
    trainImg32 = img_train[i].astype(numpy.float32)
    cuda.memcpy_htod(img_gpu,trainImg32)

    testNet.forward()

    testNet.backward()
    
    if i % batchSize == 0 or i == (len(img_train) - 1):
      testNet.optimize()      
      testNet.zero_grad()
      

  print("--- %s seconds ---" % (time.time() - start_time))
  cuda.memcpy_dtoh(training_correct,training_correct_gpu)
  reset_values(training_correct_gpu,numpy.int32(1),block=(1,1,1))
  print("correct (GPU) = ",(training_correct[0]/len(img_train)))

# --- testing ---

correct = 0
start_time = time.time()
for i in range(len(img_test)):
  testImg32 = img_test[i].astype(numpy.float32)  
  cuda.memcpy_htod(img_gpu, testImg32)

  testNet.forward()
  start = numpy.int32(testNet.layers[0] + testNet.layers[1])
  check_answer(test_correct_gpu, testNet.nodes_gpu, start, numpy.int32(label_test[i]),block=(1,1,1))
  #guess = output.index(max(output))
  #print("guess = ",guess)
print("--- %s seconds ---" % (time.time() - start_time))
cuda.memcpy_dtoh(test_correct,test_correct_gpu)
print("test dataset: correct = ",(test_correct[0]/len(img_test)))

# --------