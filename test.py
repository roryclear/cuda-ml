import pycuda.compiler as comp
import pycuda.driver as cuda
import numpy
import pycuda.autoinit

#pip install pycuda

print("ffs")

mod = comp.SourceModule(
    """
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
"""
)

multiply_them = mod.get_function("multiply_them")

a = numpy.random.randn(400).astype(numpy.float32)
b = numpy.random.randn(400).astype(numpy.float32)

dest = numpy.zeros_like(a)
multiply_them(cuda.Out(dest), cuda.In(a), cuda.In(b), block=(400, 1, 1))

print(dest - a * b)

cuda.init()
num = cuda.Device.count()
print("num = ",num)