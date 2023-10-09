import tensorflow as tf
from const import a_limit_scale, u_scale

i32 = tf.int32
f32 = tf.float32
const = tf.constant
rsum = tf.reduce_sum
rprod = tf.math.reduce_prod
rstd = tf.math.reduce_std
rmax = tf.math.reduce_max
exp = tf.exp
greater = tf.greater
less = tf.less
ln = tf.math.log
div = tf.divide
sq = tf.square
sqrt = tf.sqrt
mv = tf.linalg.matvec
mm = tf.linalg.matmul
tr = tf.linalg.trace
inv = tf.linalg.inv
pinv = tf.linalg.pinv
luinv = tf.linalg.lu_matrix_inverse
ed = tf.expand_dims
logdet = tf.linalg.logdet
det = tf.linalg.det
stack = tf.stack
pstack = tf.parallel_stack
split = tf.split
concat = tf.concat
shape = tf.shape
shape_n = tf.shape_n
relu = tf.nn.relu
tanh = tf.tanh
atanh = tf.math.atanh
scan = tf.scan
𝔼 = tf.reduce_mean
where = tf.where
clip = tf.clip_by_value
maximum = tf.maximum
minimum = tf.minimum
cast = tf.cast
reshape = tf.reshape
squeeze = tf.squeeze
add_n = tf.add_n
bce = tf.nn.sigmoid_cross_entropy_with_logits
zeros = tf.zeros
zeros_like = tf.zeros_like
ones = tf.ones
one_hot = tf.one_hot
no_grad = tf.stop_gradient
abs_tf = tf.abs
grad = tf.gradients
var = tf.Variable
normal = tf.random.normal
categorical = tf.random.categorical
uniform = tf.random.uniform
tile = tf.tile
sign = tf.sign
softplus = tf.math.softplus
softmax = tf.nn.softmax
log_softmax = tf.nn.log_softmax
to_tensor = tf.convert_to_tensor
gather = tf.gather
sigmoid = tf.math.sigmoid
log_sigmoid = tf.math.log_sigmoid
map_structure = tf.nest.map_structure
cos = tf.math.cos
sin = tf.math.sin

u_scale_tf = tf.constant(u_scale, tf.float32)
a_limit_scale_tf = tf.constant(a_limit_scale, tf.float32)

tanh1 = tf.constant(tanh(1.), tf.float32)

@tf.function
def inbounds(u):
  x = u_scale_tf*u
  return cast(less(rmax(abs_tf(x),-1),1.),tf.float32)

@tf.function
def activ(u):
  x = u_scale_tf*u
  return a_limit_scale_tf*tf.tanh(x)

@tf.function
def log_squash(x):
  return ln(abs_tf(x) + 1) * sign(x)

@tf.function
def log_tanh(x):
  return ln(tanh(x/2)*.9995+1.0005)

@tf.function
def truncated_lnσ(x):
  return ln(1e-2+0.09*sigmoid(x))
