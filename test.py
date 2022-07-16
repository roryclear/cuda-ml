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
        return 0

print("ffs")

#---- mnist stuff ---- 

(img_train, label_train), (img_test, label_test) = keras.datasets.mnist.load_data()

w0=numpy.empty((4,784)).astype(numpy.float32); w0.fill(1)
w1=numpy.empty((10,4)).astype(numpy.float32); w1.fill(1)
f = open("relu-weights784-4-10.txt", "r")
lines = f.readlines()[1:785]
i = 0
for line in lines:
  line = line.replace("\n","")
  array = line.split(",")
  for j in range(len(array)):
    w0[j][i] = array[j]
  i+=1

f = open("relu-weights784-4-10.txt", "r")
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

print("w1 = ",w1)

# --------

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
"""
)

MAX_THREADS_PER_BLOCK = \
    cuda.Device(0).get_attribute(pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK)

add_them = mod.get_function("multiply_them")
minus_them = mod.get_function("minus_them")
matrixMul = mod.get_function("matrixMul")

#a=numpy.empty(1024, dtype=numpy.float32); a.fill(numpy.float32(1))
a=numpy.matrix('3 2 1 5; 9 1 3 0', dtype=numpy.float32)
b=numpy.matrix('2 9 0; 1 3 5; 2 4 7; 8 1 5', dtype=numpy.float32)
d=numpy.matrix('0 0 0; 0 0 0', dtype=numpy.float32)

a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

b_gpu = cuda.mem_alloc(b.nbytes)
cuda.memcpy_htod(b_gpu, b)

d_gpu = cuda.mem_alloc(d.nbytes)
cuda.memcpy_htod(d_gpu, d)

n = 4
n_NP = numpy.int32(n)

nrA = 2 # number of rows in A
ncB = 3 # number of cols in B

#matrixMul(d_gpu,a_gpu,b_gpu,block=(4,4,1))
add_them(d_gpu, a_gpu, b_gpu, n_NP, block=(ncB,nrA,1))

cuda.memcpy_dtoh(d, d_gpu)
for i in range(len(d)):
  print(i," : ", d[i])
