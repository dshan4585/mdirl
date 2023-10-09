import numpy as np
import tensorflow as tf

import os
import sys
sys.path.append(os.path.abspath('../'))
from const import ns
from util.tf_util import *

m = tf.math.multiply
mm = tf.linalg.matmul
mv = tf.linalg.matvec
e = tf.math.exp
a = tf.math.add
bn = tf.nn.batch_normalization
mom = tf.nn.moments
no_grad = tf.stop_gradient

f1 = tf.nn.tanh
f2 = tf.nn.tanh
f3 = tf.nn.tanh
f4 = tf.nn.tanh
f5 = tf.nn.tanh
f6 = tf.nn.tanh
f7 = tf.nn.tanh

@tf.function
def soft_update(coeff,v1,v2,v3,v4,v5,v6,vt1,vt2,vt3,vt4,vt5,vt6):
  vt1.assign(coeff * v1 + (1-coeff) * vt1)
  vt2.assign(coeff * v2 + (1-coeff) * vt2)
  vt3.assign(coeff * v3 + (1-coeff) * vt3)
  vt4.assign(coeff * v4 + (1-coeff) * vt4)
  vt5.assign(coeff * v5 + (1-coeff) * vt5)
  vt6.assign(coeff * v6 + (1-coeff) * vt6)

@tf.function
def fwda(x,w1,b1,w2,b2,w3,b3):
  return a(mm(f1(a(mm(f1(a(mm(x,w1),b1)),w2),b2)),w3),b3)

@tf.function
def fwda1(x,w1,b1,w2,b2,w3,b3):
  return a(mv(w3,f1(a(mv(w2,f1(a(mv(w1,x,True),b1)),True),b2)),True),b3)

@tf.function
def fwdb(x,w1,b1,w2,b2,w3,b3):
  return a(mm(f2(a(mm(f2(a(mm(x,w1),b1)),w2),b2)),w3),b3)

@tf.function
def fwdb1(x,w1,b1,w2,b2,w3,b3):
  return a(mv(w3,f2(a(mv(w2,f2(a(mv(w1,x,True),b1)),True),b2)),True),b3)

@tf.function
def fwdc(x,w1,b1,w2,b2,w3,b3):
  return a(mm(f3(a(mm(f3(a(mm(x,w1),b1)),w2),b2)),w3),b3)

@tf.function
def fwdc1(x,w1,b1,w2,b2,w3,b3):
  return a(mv(w3,f3(a(mv(w2,f3(a(mv(w1,x,True),b1)),True),b2)),True),b3)

@tf.function
def fwdd(x,w1,b1,w2,b2,w3,b3):
  return a(mm(f4(a(mm(f4(a(mm(x,w1),b1)),w2),b2)),w3),b3)

@tf.function
def fwdd1(x,w1,b1,w2,b2,w3,b3):
  return a(mv(w3,f4(a(mv(w2,f4(a(mv(w1,x,True),b1)),True),b2)),True),b3)

@tf.function
def fwde(x,w1,b1,w2,b2,w3,b3):
  return a(mm(f5(a(mm(f5(a(mm(x,w1),b1)),w2),b2)),w3),b3)

@tf.function
def fwde1(x,w1,b1,w2,b2,w3,b3):
  return a(mv(w3,f5(a(mv(w2,f5(a(mv(w1,x,True),b1)),True),b2)),True),b3)

@tf.function
def fwdf(x,w1,b1,w2,b2,w3,b3):
  return a(mm(f6(a(mm(f6(a(mm(x,w1),b1)),w2),b2)),w3),b3)

@tf.function
def fwdf1(x,w1,b1,w2,b2,w3,b3):
  return a(mv(w3,f6(a(mv(w2,f6(a(mv(w1,x,True),b1)),True),b2)),True),b3)

@tf.function
def fwdg(x,w1,b1,w2,b2,w3,b3):
  return a(mm(f7(a(mm(f7(a(mm(x,w1),b1)),w2),b2)),w3),b3)

@tf.function
def fwdg1(x,w1,b1,w2,b2,w3,b3):
  return a(mv(w3,f7(a(mv(w2,f7(a(mv(w1,x,True),b1)),True),b2)),True),b3)

sz_py = list(reversed(range(ns)))
sz = const(sz_py,i32)
dut=[const([0]*(ns-1-i)+[1],f32)for i in sz_py]
@tf.function
def conv(x,F0,b0,c0,a0,w1,F1,b1,c1,a1,w2,F2,b2,c2,a2):
  n = shape(x)[0]
  nx = shape(x)[1]

  xx = reshape(x, [n,1,-1,1])
  bb0 = reshape(b0, [1,-1,nx,1])
  bb1 = reshape(b1, [1,-1,nx,1])
  bb2 = reshape(b2, [1,-1,nx,1])

  Fx0 = mm(ed(F0,0),xx-bb0)
  Fx1 = mm(ed(F1,0),xx-bb1)
  Fx2 = mm(ed(F2,0),xx-bb2)

  Fx0_ = mm(ed(F0,0),xx-bb0)
  Fx1_ = mm(ed(F1,0),xx-bb1)
  Fx2_ = mm(ed(F2,0),xx-bb2)

  cq0 = rsum(Fx0_*Fx0,[-2,-1]) + c0
  cq1 = rsum(Fx1_*Fx1,[-2,-1]) + c1
  cq2 = rsum(Fx2_*Fx2,[-2,-1]) + c2*0.

  h0 = cq0
  y0 = relu(h0) + softplus(a0)*(exp(minimum(0.,h0)/softplus(a0))-1.)
  h1 = mm(y0,softplus(w1)) + cq1
  y1 = relu(h1) + softplus(a1)*(exp(minimum(0.,h1)/softplus(a1))-1.)
  h2 = mm(y1,softplus(w2))*0. + cq2
  y2 = h2 + 0.*a2*(exp(minimum(0.,h2)/a2)-1.)


  output = reshape(y2,[-1]) + 0.5*rsum(sq(1e-3 * x),-1)
  return output

class Layer:
  def __call__(self, x):
    return x
  @property
  def vars(self):
    return []

class Dense(Layer):
  def __init__(self, nx, ny, std=0.05):
    w_init = np.random.randn(nx, ny).astype(np.float32)
    w_init *= std / np.sqrt(np.square(w_init).sum(axis=0, keepdims=True))
    self.w = tf.Variable(w_init, dtype=tf.float32)
    b_init = tf.zeros(ny, tf.float32)
    self.b = tf.Variable(b_init, dtype=tf.float32)

  def __call__(self, x):
    return x @ self.w + self.b
  @property
  def vars(self):
    return [self.w, self.b]
  @property
  def trainables(self):
    return self.vars

class BatchNorm(Layer):
  def __init__(self, nx):
    mean_init = tf.zeros(nx, tf.float32)
    var_init = tf.ones(nx, tf.float32)
    offset_init = tf.zeros(nx, tf.float32)
    scale_init = tf.ones(nx, tf.float32)
    self.mean = tf.Variable(mean_init, dtype=tf.float32, trainable=False)
    self.var = tf.Variable(var_init, dtype=tf.float32, trainable=False)
    self.offset = tf.Variable(offset_init, dtype=tf.float32)
    self.scale = tf.Variable(scale_init, dtype=tf.float32)
  def __call__(self, x):
    return (x-self.mean)/tf.sqrt(self.var + 1e-5) * self.scale + self.offset
  @property
  def vars(self):
    return [self.mean, self.var, self.offset, self.scale]
  @property
  def trainables(self):
    return [self.offset, self.scale]

class ConvQuad(Layer):
  def __init__(self, nx, ny, nr, std=0.5):
    F_init = np.random.normal(0.,std,(ny,nr,nx)).astype(np.float32)*np.sqrt(1/nx)
    b_init = np.random.normal(0.,std,(ny,nx)).astype(np.float32)*np.sqrt(1/nx)
    c_init = np.random.normal(0.,std,ny).astype(np.float32)*np.sqrt(1/nx)

    self.F = tf.Variable(F_init, dtype=tf.float32)
    self.b = tf.Variable(b_init, dtype=tf.float32)
    self.c = tf.Variable(c_init, dtype=tf.float32)
  @property
  def vars(self):
    return [self.F, self.b, self.c]
  @property
  def trainables(self):
    return self.vars

class PosDense(Layer):
  def __init__(self, nx, ny, std=0.05):
    w_init = np.random.normal(0.,std,(nx, ny)).astype(np.float32)*np.sqrt(1/nx) - 2.
    self.w = tf.Variable(w_init, dtype=tf.float32)
  @property
  def vars(self):
    return [self.w]
  @property
  def trainables(self):
    return self.vars

class CELU(Layer):
  def __init__(self, nx):
    a_init = tf.ones(nx, tf.float32)
    self.a = tf.Variable(a_init, dtype=tf.float32)
  def __call__(self, x):
    return relu(x) + minimum(0.,a*(exp(x/a)-1.))
  @property
  def vars(self):
    return [self.a]
  @property
  def trainables(self):
    return self.vars
