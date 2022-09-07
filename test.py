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
        self.weights = []
        self.nodes = []
        self.grads = []

    def copyGradsToDevice(self):
      cuda.memcpy_htod(grads_gpu,self.grads)
        
    def copyWeightsToDevice(self):
      cuda.memcpy_htod(weights_gpu,self.weights)

    def copyNodesToDevice(self):
      cuda.memcpy_htod(nodes_gpu,self.nodes)

    def optimize(self):
      length = layers[1] * layers[2] + layers[0] * layers[1]
      bx = length
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024
      optimize(weights_gpu, grads_gpu,learningRate, numpy.int32(length), block=(bx,1,1),grid=(gx,1))

    def zero_grad(self):
      length = layers[1] * layers[2] + layers[0] * layers[1]
      bx = length
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024
      reset_values(grads_gpu,numpy.int32(length),block=(bx,1,1),grid=(gx,1))
  
    def backward(self):
      length = len(nodesInput)
      bx = length
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024
      der_sigmoid(nodesInput_gpu,nodes_gpu, numpy.int32(length),block=(bx,1,1),grid=(gx,1))

      start = numpy.int32(layers[0] + layers[1])
      check_answer(training_correct_gpu, nodes_gpu, start, numpy.int32(label_train[i]),block=(1,1,1))

      #backward
      start = numpy.int32(layers[0] + layers[1])
      get_output_loss(outputLoss_gpu, nodes_gpu, start, numpy.int32(label_train[i]), block=(layers[2],1,1))
      
      startB = numpy.int32(layers[0] + layers[1])
      array_mulitply_minus(outputLossInput_gpu,outputLoss_gpu,nodesInput_gpu, startB,block=(len(outputLoss),1,1))

      n = 1
      n_NP = numpy.int32(n)

      bx = layers[1]
      by = len(outputLoss)
      gx = 1
      gy = 1

      bxn = bx
      byn = by

      if bx*by > 1024:
        if bx > by:
          bxn = int(1024 / by)
          gx = int(bx / bxn) + 1
          
      #int ncA, int ncB, int nrA
      startn0 = numpy.int32(layers[0])
      startD = numpy.int32(layers[0] * layers[1])
      startW = numpy.int32(0)
      #multiply_them_index2(float *nodesD, float *weights, float *nodesA, int ncA, int ncB, int nrA, int startn0, int startD, int startW)
      multiply_them_index3(grads_gpu, outputLossInput_gpu, nodes_gpu, n_NP, numpy.int32(bx), numpy.int32(10), startn0, startD, startW, block=(bxn,by,1), grid=(gx,gy)) 
      
      #backward first weights ???
      length = layers[1] * layers[2]
      bx = length
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024
      startD = numpy.int32(layers[0] * layers[1])
      startA = numpy.int32(layers[0] * layers[1])
      startB = numpy.int32(layers[0] * layers[1])
      array_mulitply(loss_gpu,weights_gpu,grads_gpu,startD,startA,startB,numpy.int32(length),block=(bx,1,1),grid=(gx,1))

      startA = numpy.int32(layers[0] * layers[1])
      length = layers[1]  
      bx = length
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024
      get_node_loss(totalErrors_gpu,loss_gpu,numpy.int32(layers[2]),startA,numpy.int32(length),block=(bx,1,1),grid=(gx,1))

      startB = numpy.int32(layers[0])
      startC = numpy.int32(0)
      get_grads(grads_gpu,totalErrors_gpu,nodesInput_gpu,nodes_gpu,startB,startC,block=(layers[0],1,1),grid=(layers[1],1))

    def forward(self):

      #copy input (n0_gpu) to nodes_gpu
      length = layers[0]
      bx = length
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024
      
      copy(nodes_gpu, img_gpu, numpy.int32(0), numpy.int32(0), numpy.int32(length), block=(bx,1,1), grid=(gx,1))

      
      n = len(w0[0]) # number of columns in A / number of rows in B
      n_NP = numpy.int32(n)
      bx = 1 # number of cols in B
      by = len(w0) # number of rows in A
      gx = 1
      gy = 1
      if by > 1024:
        gy = int(by / 1024) + 1
        by = 1024
      startn0 = numpy.int32(0)
      startn1 = numpy.int32(layers[0])
      startw = numpy.int32(0)
      multiply_them_index2(nodes_gpu, weights_gpu, nodes_gpu, n_NP, numpy.int32(bx) ,numpy.int32(len(w0)) , startn0, startn1,
                            startw, block=(bx,by,1), grid=(gx,gy))

      bx = layers[1]
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024

      sigmoid_index(nodes_gpu,numpy.int32(layers[0]),numpy.int32(layers[1]), block=(bx,1,1), grid=(gx,1))

      n = len(w1[0])
      n_NP = numpy.int32(n)

      bx = 1 # number of cols in B
      by = len(w1) # number of rows in A
      gx = 1
      gy = 1

      startn0 = numpy.int32(layers[0])
      startn1 = numpy.int32(layers[0] + layers[1])
      startW = numpy.int32(layers[0] * layers[1])
      multiply_them_index2(nodes_gpu, weights_gpu, nodes_gpu, n_NP, numpy.int32(bx), numpy.int32(len(w1)), startn0, startn1, startW, block=(bx,by,1), grid=(gx,gy))

      startA = numpy.int32(layers[0] + layers[1])
      startD = numpy.int32(0)
      length = layers[2]
      bx = length
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024

      start = numpy.int32(layers[0] + layers[1])
      length = numpy.int32(layers[2])
      bx = int(length)
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024
      sigmoid_index(nodes_gpu, start, length, block=(bx,1,1), grid=(gx,1))
      return 0

mod = comp.SourceModule(
    """
__global__ void multiply_them_index(float *nodesD, float *weights, float *nodesA, int ncA, int ncB, int nrA, int startn0, int startn1)
{
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;

  float t = 0;
  if(col < ncB && row < nrA)
  {
  for(int i = 0; i < ncA; i++){
    t += weights[(row * ncA) + i] * nodesA[startn0 + col + (i * ncB)];
  }
    nodesD[startn1 + (row * ncB) + col] = t;
  }
}

  __global__ void multiply_them_index2(float *nodesD, float *weights, float *nodesA, int ncA, int ncB, int nrA, int startn0, int startD, int startW)
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


__global__ void multiply_them_index3(float *nodesD, float *weights, float *nodesA, int ncA, int ncB, int nrA, int startn0, int startD, int startW)
{
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  float t = 0;
  if(col < ncB && row < nrA)
  {
  for(int i = 0; i < ncA; i++){
    t += weights[startW + (row * ncA) + i] * nodesA[startn0 + col + (i * ncB)];
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
  d[i] = -a[i] * b[startb + i];
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
    d[i] = 1 - o[start + i];
  } else {
    d[i] = 0 - o[start + i];
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
multiply_them_index2 = mod.get_function("multiply_them_index2")
multiply_them_index3 = mod.get_function("multiply_them_index3") #adds to result
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


#---- mnist stuff ---- 

(img_train, label_train), (img_test, label_test) = keras.datasets.mnist.load_data()

img_train = img_train / 255
img_test = img_test / 255

layers = [784,16,10]

numberOfNodes = 0
for i in range(len(layers)):
  numberOfNodes += layers[i]

nodesInput = numpy.zeros((numberOfNodes, 1),dtype=numpy.float32)
nodesInput_gpu = cuda.mem_alloc(nodesInput.nbytes)
cuda.memcpy_htod(nodesInput_gpu,nodesInput)

w0=numpy.empty((layers[1],layers[0])).astype(numpy.float32); w0.fill(1)
w1=numpy.empty((layers[2],layers[1])).astype(numpy.float32); w1.fill(1)

numberOfWeights = 0
for i in range(len(layers)-1):
  numberOfWeights += layers[i] * layers[i+1]

weights = numpy.zeros((numberOfWeights, 1),dtype=numpy.float32)
grads = numpy.zeros((numberOfWeights, 1),dtype=numpy.float32)

weightsFile = "sigmoid-untrained-weights"
#weightsFile = "sigmoid-weights"

for i in range(len(layers) - 1):
  weightsFile += str(layers[i]) + "-"
weightsFile += str(layers[len(layers)-1]) + "-1d.txt"

if exists(weightsFile):
  f = open(weightsFile, "r")
  lines = f.readlines()
  for i in range(len(lines)):
    line = lines[i].replace("\n","")
    weights[i] = line

else:
  print("no weights file was found")    
  for x in range(len(weights)):
    weights[x] = numpy.random.uniform() * (2 / numpy.sqrt(layers[0])) - 1 / numpy.sqrt(layers[0])

testNet = Net()
testNet.weights = weights
testNet.grads = grads
weights_gpu = cuda.mem_alloc(weights.nbytes)
testNet.copyWeightsToDevice()
grads_gpu = cuda.mem_alloc(grads.nbytes)
testNet.copyGradsToDevice()

nodes = numpy.zeros((numberOfNodes, 1),dtype=numpy.float32)
testNet.nodes = nodes
nodes_gpu = cuda.mem_alloc(nodes.nbytes)
testNet.copyNodesToDevice()

grads_gpu = cuda.mem_alloc(grads.nbytes)
cuda.memcpy_htod(grads_gpu,grads)
loss_gpu = cuda.mem_alloc(grads.nbytes)
cuda.memcpy_htod(loss_gpu,grads)


# --- training ---

outputLoss = numpy.zeros((layers[2]),dtype=numpy.float32)
outputLoss_gpu = cuda.mem_alloc(outputLoss.nbytes)
outputLossInput = numpy.zeros_like(outputLoss) #outputLoss * input
outputLossInput_gpu = cuda.mem_alloc(outputLossInput.nbytes)
cuda.memcpy_htod(outputLossInput_gpu,outputLossInput)

totalErrors = numpy.zeros((layers[1]),dtype=numpy.float32)
totalErrors_gpu = cuda.mem_alloc(totalErrors.nbytes)
cuda.memcpy_htod(totalErrors_gpu,totalErrors)

training_correct = numpy.zeros((1),dtype=numpy.int32)
training_correct_gpu = cuda.mem_alloc(training_correct.nbytes)
cuda.memcpy_htod(training_correct_gpu,training_correct)

test_correct = numpy.zeros((1),dtype=numpy.int32)
test_correct_gpu = cuda.mem_alloc(test_correct.nbytes)
cuda.memcpy_htod(test_correct_gpu,test_correct)

trainImg = img_train[0]
trainImg32 = trainImg.astype(numpy.float32)
img_gpu = cuda.mem_alloc(trainImg32.nbytes)

totalErrors = numpy.zeros((layers[1]),dtype=numpy.float32)

learningRate = numpy.float32(0.1)
batchSize = 1
for epoch in range(1):

  correct = 0
  start_time = time.time()
  for i in range(len(img_train)):  #x
    trainImg = img_train[i]
    trainImg32 = trainImg.astype(numpy.float32)
    cuda.memcpy_htod(img_gpu,trainImg32)

    #last weights

    testNet.forward()

    testNet.backward()
    


    if i % batchSize == 0 or i == (len(img_train) - 1):
      testNet.optimize()
      length = layers[1] * layers[2] + layers[0] * layers[1]
      bx = length
      gx = 1
      if bx > 1024:
        gx = int(bx / 1024) + 1
        bx = 1024
      
      testNet.zero_grad()
      

  print("--- %s seconds ---" % (time.time() - start_time))
  cuda.memcpy_dtoh(training_correct,training_correct_gpu)
  reset_values(training_correct_gpu,numpy.int32(1),block=(1,1,1))
  print("correct (GPU) = ",(training_correct[0]/len(img_train)))

# --- testing ---

correct = 0
start_time = time.time()
for i in range(len(img_test)):
  testImg = img_test[i]
  testImg32 = testImg.astype(numpy.float32)
  
  cuda.memcpy_htod(img_gpu, testImg32)

  testNet.forward()
  start = numpy.int32(layers[0] + layers[1])
  check_answer(test_correct_gpu, nodes_gpu, start, numpy.int32(label_test[i]),block=(1,1,1))
  #guess = output.index(max(output))
  #print("guess = ",guess)
print("--- %s seconds ---" % (time.time() - start_time))
cuda.memcpy_dtoh(test_correct,test_correct_gpu)
print("test dataset: correct = ",(test_correct[0]/len(img_test)))

# --------