import pycuda.compiler as comp
import pycuda.driver as cuda
import numpy
import pycuda.autoinit
import time

#pip install pycuda

print("ffs")

mod = comp.SourceModule(
    """
__global__ void add_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] + b[i];
}
"""
)

add_them = mod.get_function("add_them")

a=numpy.empty(1000).astype(numpy.float32); a.fill(1)
b=numpy.empty(1000).astype(numpy.float32); b.fill(1)

dest = numpy.zeros_like(a)

start_time = time.time()
for i in range(100000):
  add_them(cuda.Out(dest), cuda.In(a), cuda.In(b), block=(1000, 1, 1))
print("--- %s seconds ---" % (time.time() - start_time))

a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

b_gpu = cuda.mem_alloc(b.nbytes)
cuda.memcpy_htod(b_gpu, b)

d = numpy.zeros_like(a)
d_gpu = cuda.mem_alloc(d.nbytes)
cuda.memcpy_htod(d_gpu, d)

start_time = time.time()
for i in range(10000):
  add_them(d_gpu, a_gpu, b_gpu, block=(1000, 1, 1))
print("--- %s seconds ---" % (time.time() - start_time))

cuda.memcpy_dtoh(d, d_gpu)

cuda.init()
num = cuda.Device.count()
print("num = ",num)