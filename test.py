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
(img_train, label_train), (img_test, label_test) = keras.datasets.mnist.load_data()

numberr = numpy.int32(0)

mod = comp.SourceModule(
    """
__global__ void add_them(float *d, float *a, float *b)
{
  int row = threadIdx.y;
  int col = threadIdx.x;

  float t = 0;
  for(int i = 0; i < 2; i++){
    t += a[row * 2 + i] * b[col + 2 * i];
  }
  d[row * 2 + col] = t;
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

add_them = mod.get_function("add_them")
minus_them = mod.get_function("minus_them")
matrixMul = mod.get_function("matrixMul")

#a=numpy.empty(1024, dtype=numpy.float32); a.fill(numpy.float32(1))
a=numpy.matrix('2 0; 1 9', dtype=numpy.float32)
b=numpy.matrix('3 9; 4 7', dtype=numpy.float32)
d=numpy.zeros_like(a)

a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

b_gpu = cuda.mem_alloc(b.nbytes)
cuda.memcpy_htod(b_gpu, b)

d_gpu = cuda.mem_alloc(d.nbytes)
cuda.memcpy_htod(d_gpu, d)


#matrixMul(d_gpu,a_gpu,b_gpu,block=(4,4,1))
add_them(d_gpu, a_gpu, b_gpu, block=(2,2,1))

cuda.memcpy_dtoh(d, d_gpu)
for i in range(len(d)):
  print(i," : ", d[i])
