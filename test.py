import pycuda.compiler as comp
import pycuda.driver as cuda
import numpy
import pycuda.autoinit
import time
import math
from tensorflow import keras
from os.path import exists

#pip install pycuda
#pip install tensorflow

class Net():
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()

    def setSize(self, layers):
      self.layers = layers
      self.numberOfNodes = 0
      for i in range(len(self.layers)):
        self.numberOfNodes += self.layers[i]

      self.nodesGrad = numpy.zeros((self.numberOfNodes, 1),dtype=numpy.float32)
      self.nodes = numpy.zeros((self.numberOfNodes, 1),dtype=numpy.float32)
      self.loss = numpy.zeros((self.numberOfNodes, 1),dtype=numpy.float32)

      self.numberOfWeights = 0
      for i in range(len(self.layers)-1):
        self.numberOfWeights += self.layers[i] * self.layers[i+1]

      self.grads = numpy.zeros((self.numberOfWeights, 1),dtype=numpy.float32)
      self.weights = numpy.zeros((self.numberOfWeights, 1),dtype=numpy.float32)
      self.weightsLoss = numpy.zeros((self.numberOfWeights, 1),dtype=numpy.float32)

    def copyToDevice(self):
      self.weights_gpu = cuda.mem_alloc(self.weights.nbytes)
      cuda.memcpy_htod(self.weights_gpu,self.weights)
      self.nodes_gpu = cuda.mem_alloc(self.nodes.nbytes)
      cuda.memcpy_htod(self.nodes_gpu,self.nodes)
      self.grads_gpu = cuda.mem_alloc(self.grads.nbytes)
      cuda.memcpy_htod(self.grads_gpu,self.grads)
      self.nodesGrad_gpu = cuda.mem_alloc(self.nodesGrad.nbytes)
      cuda.memcpy_htod(self.nodesGrad_gpu,self.nodesGrad)
      self.loss_gpu = cuda.mem_alloc(self.loss.nbytes)
      cuda.memcpy_htod(self.loss_gpu,self.loss)

      self.weightsLoss_gpu = cuda.mem_alloc(self.weightsLoss.nbytes)
      cuda.memcpy_htod(self.weightsLoss_gpu,self.weightsLoss)

    def loadWeights(self, path):
      weightsFile = path
      for i in range(len(self.layers) - 1):
        weightsFile += str(self.layers[i]) + "-"
      weightsFile += str(self.layers[len(self.layers)-1]) + ".txt"
      if exists(weightsFile):
        f = open(weightsFile, "r")
        lines = f.readlines()
        for i in range(len(lines)):
          line = lines[i].replace("\n","")
          self.weights[i] = line

      else:
        print("no weights file was found")    
        for x in range(len(self.weights)):
          self.weights[x] = numpy.random.uniform() * (2 / numpy.sqrt(self.layers[0])) - 1 / numpy.sqrt(self.layers[0])

    def optimize(self):
      length = len(self.weights)
      bx,by,gx,gy = self.getBlockAndGridSize(length,1)
      optimize(self.weights_gpu, self.grads_gpu,self.learningRate, numpy.int32(length), block=(bx,by,1),grid=(gx,gy))

    def zero_grad(self):
      length = len(self.weights)
      bx,by,gx,gy = self.getBlockAndGridSize(length,1)
      reset_values(self.grads_gpu,numpy.int32(length),block=(bx,by,1),grid=(gx,gy))
  
    def backward(self):
      length = len(self.nodesGrad)

      bx,by,gx,gy = self.getBlockAndGridSize(length,1)

      der_sigmoid(self.nodesGrad_gpu,self.nodes_gpu, numpy.int32(length),block=(bx,by,1),grid=(gx,gy))

      numberOfLayers = len(self.layers)

      startw0 = numpy.int32(len(self.weights) - (self.layers[numberOfLayers-1] * self.layers[numberOfLayers-2]))
      startn1 = numpy.int32(self.numberOfNodes - self.layers[numberOfLayers-1])
      startn0 = startn1 - numpy.int32(self.layers[numberOfLayers-2])
      lengthn0 = self.layers[numberOfLayers-2]
      lengthn1 = self.layers[numberOfLayers-1]
      lengthw1 = self.layers[numberOfLayers-2] * self.layers[numberOfLayers-1]

      ###---------------------------

      start = startn1
      check_answer(training_correct_gpu, self.nodes_gpu, start, numpy.int32(label_train[i]),block=(1,1,1))

      #backward
      start = startn1
      lengthx = lengthn1
      lengthy = 1

      bx,by,gx,gy = self.getBlockAndGridSize(lengthx,lengthy)

      get_output_loss(self.loss_gpu, self.nodes_gpu, start, numpy.int32(label_train[i]),
                      block=(bx,by,1),grid=(gx,gy))
      
      lengthx = lengthn0
      lengthy = lengthn1

      bx,by,gx,gy = self.getBlockAndGridSize(lengthx,lengthy)

      #int ncA, int ncB, int nrA
      startC = startn0
      startD = startw0
      startA = startn1
      startB = startA
      ncB = numpy.int32(lengthn0)
      nrA = numpy.int32(lengthn1)
      #__global__ void multiply_them_index_minus(float *d, float *a, float *b ,float *c, int startA, int startB, int startC, int startD, int ncB, int nrA)
      multiply_them_index_add(self.grads_gpu, self.loss_gpu, self.nodesGrad_gpu,
       self.nodes_gpu, startA, startB, startC, startD, ncB, nrA,
        block=(bx,by,1), grid=(gx,gy)) 
      
      #backward first weights ???
      startw1 = numpy.int32(len(self.weights))
      for y in range(len(self.layers)-2):

        startw1 -= numpy.int32(self.layers[numberOfLayers-1-y] * self.layers[numberOfLayers-2-y])
        startw0 -= numpy.int32(self.layers[numberOfLayers-2-y] * self.layers[numberOfLayers-3-y])

        startn1 -= numpy.int32(self.layers[numberOfLayers-2-y])
        startn0 -= numpy.int32(self.layers[numberOfLayers-3-y])

        lengthn0 = self.layers[numberOfLayers-3-y]
        lengthn1 = self.layers[numberOfLayers-2-y]
        lengthn2 = self.layers[numberOfLayers-1-y]

        lengthw1 = self.layers[numberOfLayers-3-y] * self.layers[numberOfLayers-2-y]
        #print("lengthw1 =",lengthw1)

        length = lengthw1
        bx = length
        gx = 1
        if bx > MAX_THREADS_PER_BLOCK:
          gx = int(bx / MAX_THREADS_PER_BLOCK) + 1
          bx = MAX_THREADS_PER_BLOCK
        startD = startn1
        startA = startw1
        startB = startw1
        array_mulitply(self.weightsLoss_gpu,self.weights_gpu,self.grads_gpu,startD,startA,startB,numpy.int32(length)
        ,block=(bx,1,1),grid=(gx,1))

        startA = startn1
        length = lengthn1
        startD = startn0
        bx = length
        gx = 1
        if bx > MAX_THREADS_PER_BLOCK:
          gx = int(bx / MAX_THREADS_PER_BLOCK) + 1
          bx = MAX_THREADS_PER_BLOCK
        numberOfNodesInLayer = numpy.int32(lengthn2)
        get_node_loss(self.loss_gpu,self.weightsLoss_gpu,numberOfNodesInLayer,startA,startD,
                      numpy.int32(length),block=(bx,1,1),grid=(gx,1))

        startA = startn0
        startB = startn1
        startC = startn0
        startD = startw0

        lengthx = lengthn0
        lengthy = lengthn1

        bx,by,gx,gy = self.getBlockAndGridSize(lengthx,lengthy)

        multiply_them_index_add(self.grads_gpu,self.loss_gpu,self.nodesGrad_gpu, self.nodes_gpu,startA,startB,startC,startD,numpy.int32(lengthx),numpy.int32(lengthy),
                  block=(bx,by,1),grid=(gx,gy))

    def forward(self, input):

      #copy input (n0_gpu) to nodes_gpu
      length = self.layers[0]

      bx,by,gx,gy = self.getBlockAndGridSize(length,1)
  
      copy(self.nodes_gpu, input, numpy.int32(0), numpy.int32(0), numpy.int32(length), block=(bx,by,1), grid=(gx,gy))

      startn0 = numpy.int32(0)
      startn1 = numpy.int32(self.layers[0])
      startw = numpy.int32(0)
      start = numpy.int32(0)
      for x in range(len(self.layers)-1):
        
        if x > 0:
          startw += numpy.int32(self.layers[x-1] * self.layers[x])
          startn1 += numpy.int32(self.layers[x])
          startn0 += numpy.int32(self.layers[x-1])


        n = self.layers[x] # number of columns in A / number of rows in B
        n_NP = numpy.int32(n)
        nrA = numpy.int32(self.layers[x+1])

        bx,by,gx,gy = self.getBlockAndGridSize(1,self.layers[x+1]) # number of cols in B, number of rows in A

        multiply_them_index(self.nodes_gpu, self.weights_gpu, self.nodes_gpu, n_NP, numpy.int32(bx) 
        ,nrA , startn0, startn1,
                              startw, block=(bx,by,1), grid=(gx,gy))

        length = self.layers[x+1]
        start += numpy.int32(self.layers[x])

        bx,by,gx,gy = self.getBlockAndGridSize(length,1)

        sigmoid_index(self.nodes_gpu,start,numpy.int32(length),
                      block=(bx,by,1), grid=(gx,gy))


      return 0

    def getBlockAndGridSize(self,lengthx,lengthy):
      bx = lengthx
      by = lengthy
      gx = 1
      gy = 1
      if bx > MAX_THREADS_PER_BLOCK:
        gx = math.ceil(bx / MAX_THREADS_PER_BLOCK)
        bx = MAX_THREADS_PER_BLOCK

      if by > MAX_THREADS_PER_BLOCK:
        gy = math.ceil(by / MAX_THREADS_PER_BLOCK)
        by = MAX_THREADS_PER_BLOCK

      if bx * by > MAX_THREADS_PER_BLOCK:
        if by > bx:
          bx = math.ceil(MAX_THREADS_PER_BLOCK / by)
          gx = math.ceil(lengthx / bx)
        else:
          by = int(MAX_THREADS_PER_BLOCK / bx)
          gy = math.ceil(lengthy / by) 
      return bx,by,gx,gy

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


__global__ void multiply_them_index_add(float *d, float *a, float *b ,float *c, int startA, int startB, int startC, int startD, int ncB, int nrA)
{
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  if(col < ncB && row < nrA)
  {
    d[startD + (row * ncB) + col] += a[startA + row] * b[startB + row] * c[startC + col];
  }
}

__global__ void optimize(float *d, float *a, float lr, int length)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < length)
  {
  d[i] = (lr * -a[i]) + d[i];
  }
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
    d[start + i] = o[start + i] - 1;
  } else {
    d[start + i] = o[start + i];
  }
}

__global__ void get_node_loss(float *d, float *a, int n, int startA, int startD, int length)
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
  d[startD + i] = t / n;
  }
}

__global__ void reset_values(float *d, int length)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < length)
  {
    d[i] = 0;
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

multiply_them_index = mod.get_function("multiply_them_index")
multiply_them_index_add = mod.get_function("multiply_them_index_add") #adds to result
optimize = mod.get_function("optimize")
sigmoid_index = mod.get_function("sigmoid_index")
der_sigmoid = mod.get_function("der_sigmoid")
array_mulitply = mod.get_function("array_mulitply")
get_output_loss = mod.get_function("get_output_loss")
get_node_loss = mod.get_function("get_node_loss")
reset_values = mod.get_function("reset_values")
check_answer = mod.get_function("check_answer")
copy = mod.get_function("copy")

def test(testNet):
  reset_values(test_correct_gpu,numpy.int32(1),block=(1,1,1))
  start_time = time.time()
  start = numpy.int32(0)
  for x in range(len(testNet.layers)-1):
    start += numpy.int32(testNet.layers[x])
  for i in range(len(img_test)):
    testImg32 = img_test[i].astype(numpy.float32)  
    cuda.memcpy_htod(img_gpu, testImg32)

    testNet.forward(img_gpu)
    check_answer(test_correct_gpu, testNet.nodes_gpu, start, numpy.int32(label_test[i]),block=(1,1,1))
  print("--- %s seconds ---" % (time.time() - start_time))
  cuda.memcpy_dtoh(test_correct,test_correct_gpu)
  print("test dataset: correct = ",(test_correct[0]/len(img_test)))

#---- mnist stuff ---- 

(img_train, label_train), (img_test, label_test) = keras.datasets.mnist.load_data()

img_train = img_train / 255
img_test = img_test / 255

training_correct = numpy.zeros((1),dtype=numpy.int32)
training_correct_gpu = cuda.mem_alloc(training_correct.nbytes)
cuda.memcpy_htod(training_correct_gpu,training_correct)

test_correct = numpy.zeros((1),dtype=numpy.int32)
test_correct_gpu = cuda.mem_alloc(test_correct.nbytes)
cuda.memcpy_htod(test_correct_gpu,test_correct)

trainImg32 = img_train[0].astype(numpy.float32)
img_gpu = cuda.mem_alloc(trainImg32.nbytes)