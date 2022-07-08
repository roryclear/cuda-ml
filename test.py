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
        self.weights = [[]]
        
    def forward(self, input):
        return 0

print("ffs")
(img_train, label_train), (img_test, label_test) = keras.datasets.mnist.load_data()

mod = comp.SourceModule(
    """
__global__ void add_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] + b[i];
}

__global__ void relu(float *a)
{
  const int i = threadIdx.x;
  if(a[i] < 0) 
  {
    a[i] = 0;
  }
}
"""
)

add_them = mod.get_function("add_them")

a=numpy.empty(1000).astype(numpy.float32); a.fill(1)
b=numpy.empty(1000).astype(numpy.float32); b.fill(1)

w1=numpy.empty((4,784)).astype(numpy.float32); w1.fill(1)
print("w1 = ",w1)
f = open("relu-weights784-4-10.txt", "r")
lines = f.readlines()[1:785]
for line in lines:
  line = line.replace("\n","")
  array = line.split(",")
  print(array)

dest = numpy.zeros_like(a)

start_time = time.time()
for i in range(100000):
  add_them(cuda.Out(a), cuda.In(a), cuda.In(b), block=(1000, 1, 1))
print("--- %s seconds ---" % (time.time() - start_time))

print(a[0])

a=numpy.empty(1000).astype(numpy.float32); a.fill(1)
b=numpy.empty(1000).astype(numpy.float32); b.fill(1)

a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

b_gpu = cuda.mem_alloc(b.nbytes)
cuda.memcpy_htod(b_gpu, b)

d = numpy.zeros_like(a)
d_gpu = cuda.mem_alloc(d.nbytes)
cuda.memcpy_htod(d_gpu, d)

start_time = time.time()
for i in range(10000):
  add_them(a_gpu, a_gpu, b_gpu, block=(1000, 1, 1))
print("--- %s seconds ---" % (time.time() - start_time))

cuda.memcpy_dtoh(d, a_gpu)

print(d[0])

cuda.init()