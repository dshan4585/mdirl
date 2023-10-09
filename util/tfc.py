import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # NO DEBUGGING
os.environ['CUDA_VISIBLE_DEVICES']='' # NO GPU
import platform
from itertools import accumulate
import tensorflow as tf

tf.config.set_soft_device_placement(True)

nw = 1
rank = 0

n_cpu = os.cpu_count()
if platform.system() == 'Darwin': # MacOS
  n_cpu //= 2

if nw == 1:
  n_parallel = 1
if n_cpu <= nw:
  n_parallel = 1
else:
  n_parallel = 1

tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

def allocate_gpu(n_gpu, nw, i):
  assert 0 <= i < nw
  if n_gpu <= nw:
    a = nw // n_gpu
    b = nw % n_gpu
    nw_accums = list(accumulate([a +
        (1 if i < b else 0)
        for i in range(n_gpu)]))
    for gpu_idx, nw_accum in enumerate(nw_accums):
      if i < nw_accum:
        break
    return gpu_idx
  else:
    return i

gpus = tf.config.list_physical_devices('GPU')

if gpus:
  n_gpu = len(gpus)
  gpu_idx = allocate_gpu(n_gpu, nw, rank)

  alloc_gpus = [gpus[gpu_idx]]
  tf.config.set_visible_devices(alloc_gpus, 'GPU')
  for gpu in alloc_gpus:
    tf.config.experimental.set_memory_growth(gpu, True)