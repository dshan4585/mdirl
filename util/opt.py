import tensorflow as tf
import numpy as np

def set_flat1(flat,flat_sizes,v1):
  f1 = flat
  v1.assign(tf.reshape(f1,tf.shape(v1)))
def set_flat2(flat,flat_sizes,v1,v2):
  f1,f2 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
def set_flat6(flat,flat_sizes,v1,v2,v3,v4,v5,v6):
  f1,f2,f3,f4,f5,f6 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))

def set_flat7(flat,flat_sizes,v1,v2,v3,v4,v5,v6,v7):
  f1,f2,f3,f4,f5,f6,f7 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))

def set_flat8(flat,flat_sizes,v1,v2,v3,v4,v5,v6,v7,v8):
  f1,f2,f3,f4,f5,f6,f7,f8 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))

def set_flat10(flat,flat_sizes,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))

def set_flat12(flat,flat_sizes,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))

def set_flat13(flat,flat_sizes,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))

def set_flat18(flat,flat_sizes,
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    v11,v12,v13,v14,v15,v16,v17,v18):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))
  v14.assign(tf.reshape(f14,tf.shape(v14)))
  v15.assign(tf.reshape(f15,tf.shape(v15)))
  v16.assign(tf.reshape(f16,tf.shape(v16)))
  v17.assign(tf.reshape(f17,tf.shape(v17)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))

def set_flat19(flat,flat_sizes,
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    v11,v12,v13,v14,v15,v16,v17,v18,v19):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))
  v14.assign(tf.reshape(f14,tf.shape(v14)))
  v15.assign(tf.reshape(f15,tf.shape(v15)))
  v16.assign(tf.reshape(f16,tf.shape(v16)))
  v17.assign(tf.reshape(f17,tf.shape(v17)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))
  v19.assign(tf.reshape(f19,tf.shape(v19)))

def set_flat20(flat,flat_sizes,
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    v11,v12,v13,v14,v15,v16,v17,v18,v19,v20):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))
  v14.assign(tf.reshape(f14,tf.shape(v14)))
  v15.assign(tf.reshape(f15,tf.shape(v15)))
  v16.assign(tf.reshape(f16,tf.shape(v16)))
  v17.assign(tf.reshape(f17,tf.shape(v17)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))
  v19.assign(tf.reshape(f19,tf.shape(v19)))
  v20.assign(tf.reshape(f20,tf.shape(v20)))


def set_flat24(flat,flat_sizes,
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
    v21,v22,v23,v24):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))
  v14.assign(tf.reshape(f14,tf.shape(v14)))
  v15.assign(tf.reshape(f15,tf.shape(v15)))
  v16.assign(tf.reshape(f16,tf.shape(v16)))
  v17.assign(tf.reshape(f17,tf.shape(v17)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))
  v19.assign(tf.reshape(f19,tf.shape(v19)))
  v20.assign(tf.reshape(f20,tf.shape(v20)))
  v21.assign(tf.reshape(f21,tf.shape(v21)))
  v22.assign(tf.reshape(f22,tf.shape(v22)))
  v23.assign(tf.reshape(f23,tf.shape(v23)))
  v24.assign(tf.reshape(f24,tf.shape(v24)))

def set_flat25(flat,flat_sizes,
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
    v21,v22,v23,v24,v25):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))
  v14.assign(tf.reshape(f14,tf.shape(v14)))
  v15.assign(tf.reshape(f15,tf.shape(v15)))
  v16.assign(tf.reshape(f16,tf.shape(v16)))
  v17.assign(tf.reshape(f17,tf.shape(v17)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))
  v19.assign(tf.reshape(f19,tf.shape(v19)))
  v20.assign(tf.reshape(f20,tf.shape(v20)))
  v21.assign(tf.reshape(f21,tf.shape(v21)))
  v22.assign(tf.reshape(f22,tf.shape(v22)))
  v23.assign(tf.reshape(f23,tf.shape(v23)))
  v24.assign(tf.reshape(f24,tf.shape(v24)))
  v24.assign(tf.reshape(f24,tf.shape(v24)))
  v25.assign(tf.reshape(f25,tf.shape(v25)))

def set_flat30(flat,flat_sizes,
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
    v21,v22,v23,v24,v25,v26,v27,v28,v29,v30):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))
  v14.assign(tf.reshape(f14,tf.shape(v14)))
  v15.assign(tf.reshape(f15,tf.shape(v15)))
  v16.assign(tf.reshape(f16,tf.shape(v16)))
  v17.assign(tf.reshape(f17,tf.shape(v17)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))
  v19.assign(tf.reshape(f19,tf.shape(v19)))
  v20.assign(tf.reshape(f20,tf.shape(v20)))
  v21.assign(tf.reshape(f21,tf.shape(v21)))
  v22.assign(tf.reshape(f22,tf.shape(v22)))
  v23.assign(tf.reshape(f23,tf.shape(v23)))
  v24.assign(tf.reshape(f24,tf.shape(v24)))
  v25.assign(tf.reshape(f25,tf.shape(v25)))
  v26.assign(tf.reshape(f26,tf.shape(v26)))
  v27.assign(tf.reshape(f27,tf.shape(v27)))
  v28.assign(tf.reshape(f28,tf.shape(v28)))
  v29.assign(tf.reshape(f29,tf.shape(v29)))
  v30.assign(tf.reshape(f30,tf.shape(v30)))

def set_flat31(flat,flat_sizes,
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
    v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))
  v14.assign(tf.reshape(f14,tf.shape(v14)))
  v15.assign(tf.reshape(f15,tf.shape(v15)))
  v16.assign(tf.reshape(f16,tf.shape(v16)))
  v17.assign(tf.reshape(f17,tf.shape(v17)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))
  v19.assign(tf.reshape(f19,tf.shape(v19)))
  v20.assign(tf.reshape(f20,tf.shape(v20)))
  v21.assign(tf.reshape(f21,tf.shape(v21)))
  v22.assign(tf.reshape(f22,tf.shape(v22)))
  v23.assign(tf.reshape(f23,tf.shape(v23)))
  v24.assign(tf.reshape(f24,tf.shape(v24)))
  v25.assign(tf.reshape(f25,tf.shape(v25)))
  v26.assign(tf.reshape(f26,tf.shape(v26)))
  v27.assign(tf.reshape(f27,tf.shape(v27)))
  v28.assign(tf.reshape(f28,tf.shape(v28)))
  v29.assign(tf.reshape(f29,tf.shape(v29)))
  v30.assign(tf.reshape(f30,tf.shape(v30)))
  v31.assign(tf.reshape(f31,tf.shape(v31)))

def set_flat34(flat,flat_sizes,
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
    v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33,v34):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))
  v14.assign(tf.reshape(f14,tf.shape(v14)))
  v15.assign(tf.reshape(f15,tf.shape(v15)))
  v16.assign(tf.reshape(f16,tf.shape(v16)))
  v17.assign(tf.reshape(f17,tf.shape(v17)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))
  v19.assign(tf.reshape(f19,tf.shape(v19)))
  v20.assign(tf.reshape(f20,tf.shape(v20)))
  v21.assign(tf.reshape(f21,tf.shape(v21)))
  v22.assign(tf.reshape(f22,tf.shape(v22)))
  v23.assign(tf.reshape(f23,tf.shape(v23)))
  v24.assign(tf.reshape(f24,tf.shape(v24)))
  v25.assign(tf.reshape(f25,tf.shape(v25)))
  v26.assign(tf.reshape(f26,tf.shape(v26)))
  v27.assign(tf.reshape(f27,tf.shape(v27)))
  v28.assign(tf.reshape(f28,tf.shape(v28)))
  v29.assign(tf.reshape(f29,tf.shape(v29)))
  v30.assign(tf.reshape(f30,tf.shape(v30)))
  v31.assign(tf.reshape(f31,tf.shape(v31)))
  v32.assign(tf.reshape(f32,tf.shape(v32)))
  v33.assign(tf.reshape(f33,tf.shape(v33)))
  v34.assign(tf.reshape(f34,tf.shape(v34)))

def set_flat35(flat,flat_sizes,
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
    v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33,v34,v35):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))
  v14.assign(tf.reshape(f14,tf.shape(v14)))
  v15.assign(tf.reshape(f15,tf.shape(v15)))
  v16.assign(tf.reshape(f16,tf.shape(v16)))
  v17.assign(tf.reshape(f17,tf.shape(v17)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))
  v19.assign(tf.reshape(f19,tf.shape(v19)))
  v20.assign(tf.reshape(f20,tf.shape(v20)))
  v21.assign(tf.reshape(f21,tf.shape(v21)))
  v22.assign(tf.reshape(f22,tf.shape(v22)))
  v23.assign(tf.reshape(f23,tf.shape(v23)))
  v24.assign(tf.reshape(f24,tf.shape(v24)))
  v25.assign(tf.reshape(f25,tf.shape(v25)))
  v26.assign(tf.reshape(f26,tf.shape(v26)))
  v27.assign(tf.reshape(f27,tf.shape(v27)))
  v28.assign(tf.reshape(f28,tf.shape(v28)))
  v29.assign(tf.reshape(f29,tf.shape(v29)))
  v30.assign(tf.reshape(f30,tf.shape(v30)))
  v31.assign(tf.reshape(f31,tf.shape(v31)))
  v32.assign(tf.reshape(f32,tf.shape(v32)))
  v33.assign(tf.reshape(f33,tf.shape(v33)))
  v34.assign(tf.reshape(f34,tf.shape(v34)))
  v35.assign(tf.reshape(f35,tf.shape(v35)))

def set_flat36(flat,flat_sizes,
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
    v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33,v34,v35,v36):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))
  v14.assign(tf.reshape(f14,tf.shape(v14)))
  v15.assign(tf.reshape(f15,tf.shape(v15)))
  v16.assign(tf.reshape(f16,tf.shape(v16)))
  v17.assign(tf.reshape(f17,tf.shape(v17)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))
  v19.assign(tf.reshape(f19,tf.shape(v19)))
  v20.assign(tf.reshape(f20,tf.shape(v20)))
  v21.assign(tf.reshape(f21,tf.shape(v21)))
  v22.assign(tf.reshape(f22,tf.shape(v22)))
  v23.assign(tf.reshape(f23,tf.shape(v23)))
  v24.assign(tf.reshape(f24,tf.shape(v24)))
  v25.assign(tf.reshape(f25,tf.shape(v25)))
  v26.assign(tf.reshape(f26,tf.shape(v26)))
  v27.assign(tf.reshape(f27,tf.shape(v27)))
  v28.assign(tf.reshape(f28,tf.shape(v28)))
  v29.assign(tf.reshape(f29,tf.shape(v29)))
  v30.assign(tf.reshape(f30,tf.shape(v30)))
  v31.assign(tf.reshape(f31,tf.shape(v31)))
  v32.assign(tf.reshape(f32,tf.shape(v32)))
  v33.assign(tf.reshape(f33,tf.shape(v33)))
  v34.assign(tf.reshape(f34,tf.shape(v34)))
  v35.assign(tf.reshape(f35,tf.shape(v35)))
  v36.assign(tf.reshape(f36,tf.shape(v36)))

def set_flat37(flat,flat_sizes,
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
    v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33,v34,v35,v36,v37):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36,f37 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))
  v14.assign(tf.reshape(f14,tf.shape(v14)))
  v15.assign(tf.reshape(f15,tf.shape(v15)))
  v16.assign(tf.reshape(f16,tf.shape(v16)))
  v17.assign(tf.reshape(f17,tf.shape(v17)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))
  v19.assign(tf.reshape(f19,tf.shape(v19)))
  v20.assign(tf.reshape(f20,tf.shape(v20)))
  v21.assign(tf.reshape(f21,tf.shape(v21)))
  v22.assign(tf.reshape(f22,tf.shape(v22)))
  v23.assign(tf.reshape(f23,tf.shape(v23)))
  v24.assign(tf.reshape(f24,tf.shape(v24)))
  v25.assign(tf.reshape(f25,tf.shape(v25)))
  v26.assign(tf.reshape(f26,tf.shape(v26)))
  v27.assign(tf.reshape(f27,tf.shape(v27)))
  v28.assign(tf.reshape(f28,tf.shape(v28)))
  v29.assign(tf.reshape(f29,tf.shape(v29)))
  v30.assign(tf.reshape(f30,tf.shape(v30)))
  v31.assign(tf.reshape(f31,tf.shape(v31)))
  v32.assign(tf.reshape(f32,tf.shape(v32)))
  v33.assign(tf.reshape(f33,tf.shape(v33)))
  v34.assign(tf.reshape(f34,tf.shape(v34)))
  v35.assign(tf.reshape(f35,tf.shape(v35)))
  v36.assign(tf.reshape(f36,tf.shape(v36)))
  v37.assign(tf.reshape(f37,tf.shape(v37)))

def set_flat42(flat,flat_sizes,
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
    v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33,v34,v35,v36,v37,v38,v39,v40,v41,v42):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36,f37,f38,f39,f40,f41,f42 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))
  v14.assign(tf.reshape(f14,tf.shape(v14)))
  v15.assign(tf.reshape(f15,tf.shape(v15)))
  v16.assign(tf.reshape(f16,tf.shape(v16)))
  v17.assign(tf.reshape(f17,tf.shape(v17)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))
  v19.assign(tf.reshape(f19,tf.shape(v19)))
  v20.assign(tf.reshape(f20,tf.shape(v20)))
  v21.assign(tf.reshape(f21,tf.shape(v21)))
  v22.assign(tf.reshape(f22,tf.shape(v22)))
  v23.assign(tf.reshape(f23,tf.shape(v23)))
  v24.assign(tf.reshape(f24,tf.shape(v24)))
  v25.assign(tf.reshape(f25,tf.shape(v25)))
  v26.assign(tf.reshape(f26,tf.shape(v26)))
  v27.assign(tf.reshape(f27,tf.shape(v27)))
  v28.assign(tf.reshape(f28,tf.shape(v28)))
  v29.assign(tf.reshape(f29,tf.shape(v29)))
  v30.assign(tf.reshape(f30,tf.shape(v30)))
  v31.assign(tf.reshape(f31,tf.shape(v31)))
  v32.assign(tf.reshape(f32,tf.shape(v32)))
  v33.assign(tf.reshape(f33,tf.shape(v33)))
  v34.assign(tf.reshape(f34,tf.shape(v34)))
  v35.assign(tf.reshape(f35,tf.shape(v35)))
  v36.assign(tf.reshape(f36,tf.shape(v36)))
  v37.assign(tf.reshape(f37,tf.shape(v37)))
  v38.assign(tf.reshape(f38,tf.shape(v38)))
  v39.assign(tf.reshape(f39,tf.shape(v39)))
  v40.assign(tf.reshape(f40,tf.shape(v40)))
  v41.assign(tf.reshape(f41,tf.shape(v41)))
  v42.assign(tf.reshape(f42,tf.shape(v42)))

def set_flat48(flat,flat_sizes,
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
    v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33,v34,v35,v36,v37,v38,v39,v40,v41,v42,v43,v44,v45,v46,v47,v48):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36,f37,f38,f39,f40,f41,f42,f43,f44,f45,f46,f47,f48 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))
  v14.assign(tf.reshape(f14,tf.shape(v14)))
  v15.assign(tf.reshape(f15,tf.shape(v15)))
  v16.assign(tf.reshape(f16,tf.shape(v16)))
  v17.assign(tf.reshape(f17,tf.shape(v17)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))
  v19.assign(tf.reshape(f19,tf.shape(v19)))
  v20.assign(tf.reshape(f20,tf.shape(v20)))
  v21.assign(tf.reshape(f21,tf.shape(v21)))
  v22.assign(tf.reshape(f22,tf.shape(v22)))
  v23.assign(tf.reshape(f23,tf.shape(v23)))
  v24.assign(tf.reshape(f24,tf.shape(v24)))
  v25.assign(tf.reshape(f25,tf.shape(v25)))
  v26.assign(tf.reshape(f26,tf.shape(v26)))
  v27.assign(tf.reshape(f27,tf.shape(v27)))
  v28.assign(tf.reshape(f28,tf.shape(v28)))
  v29.assign(tf.reshape(f29,tf.shape(v29)))
  v30.assign(tf.reshape(f30,tf.shape(v30)))
  v31.assign(tf.reshape(f31,tf.shape(v31)))
  v32.assign(tf.reshape(f32,tf.shape(v32)))
  v33.assign(tf.reshape(f33,tf.shape(v33)))
  v34.assign(tf.reshape(f34,tf.shape(v34)))
  v35.assign(tf.reshape(f35,tf.shape(v35)))
  v36.assign(tf.reshape(f36,tf.shape(v36)))
  v37.assign(tf.reshape(f37,tf.shape(v37)))
  v38.assign(tf.reshape(f38,tf.shape(v38)))
  v39.assign(tf.reshape(f39,tf.shape(v39)))
  v40.assign(tf.reshape(f40,tf.shape(v40)))
  v41.assign(tf.reshape(f41,tf.shape(v41)))
  v42.assign(tf.reshape(f42,tf.shape(v42)))
  v43.assign(tf.reshape(f43,tf.shape(v43)))
  v44.assign(tf.reshape(f44,tf.shape(v44)))
  v45.assign(tf.reshape(f45,tf.shape(v45)))
  v46.assign(tf.reshape(f46,tf.shape(v46)))
  v47.assign(tf.reshape(f47,tf.shape(v47)))
  v48.assign(tf.reshape(f48,tf.shape(v48)))

def set_flat58(flat,flat_sizes,
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
    v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,
    v31,v32,v33,v34,v35,v36,v37,v38,v39,v40,
    v41,v42,v43,v44,v45,v46,v47,v48,v49,v50,
    v51,v52,v53,v54,v55,v56,v57,v58):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,\
  f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,\
  f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,\
  f31,f32,f33,f34,f35,f36,f37,f38,f39,f40,\
  f41,f42,f43,f44,f45,f46,f47,f48,f49,f50,\
  f51,f52,f53,f54,f55,f56,f57,f58 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))
  v14.assign(tf.reshape(f14,tf.shape(v14)))
  v15.assign(tf.reshape(f15,tf.shape(v15)))
  v16.assign(tf.reshape(f16,tf.shape(v16)))
  v17.assign(tf.reshape(f17,tf.shape(v17)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))
  v19.assign(tf.reshape(f19,tf.shape(v19)))
  v20.assign(tf.reshape(f20,tf.shape(v20)))
  v21.assign(tf.reshape(f21,tf.shape(v21)))
  v22.assign(tf.reshape(f22,tf.shape(v22)))
  v23.assign(tf.reshape(f23,tf.shape(v23)))
  v24.assign(tf.reshape(f24,tf.shape(v24)))
  v25.assign(tf.reshape(f25,tf.shape(v25)))
  v26.assign(tf.reshape(f26,tf.shape(v26)))
  v27.assign(tf.reshape(f27,tf.shape(v27)))
  v28.assign(tf.reshape(f28,tf.shape(v28)))
  v29.assign(tf.reshape(f29,tf.shape(v29)))
  v30.assign(tf.reshape(f30,tf.shape(v30)))
  v31.assign(tf.reshape(f31,tf.shape(v31)))
  v32.assign(tf.reshape(f32,tf.shape(v32)))
  v33.assign(tf.reshape(f33,tf.shape(v33)))
  v34.assign(tf.reshape(f34,tf.shape(v34)))
  v35.assign(tf.reshape(f35,tf.shape(v35)))
  v36.assign(tf.reshape(f36,tf.shape(v36)))
  v37.assign(tf.reshape(f37,tf.shape(v37)))
  v38.assign(tf.reshape(f38,tf.shape(v38)))
  v39.assign(tf.reshape(f39,tf.shape(v39)))
  v40.assign(tf.reshape(f40,tf.shape(v40)))
  v41.assign(tf.reshape(f41,tf.shape(v41)))
  v42.assign(tf.reshape(f42,tf.shape(v42)))
  v43.assign(tf.reshape(f43,tf.shape(v43)))
  v44.assign(tf.reshape(f44,tf.shape(v44)))
  v45.assign(tf.reshape(f45,tf.shape(v45)))
  v46.assign(tf.reshape(f46,tf.shape(v46)))
  v47.assign(tf.reshape(f47,tf.shape(v47)))
  v48.assign(tf.reshape(f48,tf.shape(v48)))
  v49.assign(tf.reshape(f49,tf.shape(v49)))
  v50.assign(tf.reshape(f50,tf.shape(v50)))
  v51.assign(tf.reshape(f51,tf.shape(v51)))
  v52.assign(tf.reshape(f52,tf.shape(v52)))
  v53.assign(tf.reshape(f53,tf.shape(v53)))
  v54.assign(tf.reshape(f54,tf.shape(v54)))
  v55.assign(tf.reshape(f55,tf.shape(v55)))
  v56.assign(tf.reshape(f56,tf.shape(v56)))
  v57.assign(tf.reshape(f57,tf.shape(v57)))
  v58.assign(tf.reshape(f58,tf.shape(v58)))

def set_flat64(flat,flat_sizes,
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
    v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,
    v31,v32,v33,v34,v35,v36,v37,v38,v39,v40,
    v41,v42,v43,v44,v45,v46,v47,v48,v49,v50,
    v51,v52,v53,v54,v55,v56,v57,v58,v59,v60,
    v61,v62,v63,v64):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,\
  f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,\
  f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,\
  f31,f32,f33,f34,f35,f36,f37,f38,f39,f40,\
  f41,f42,f43,f44,f45,f46,f47,f48,f49,f50,\
  f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,\
  f61,f62,f63,f64 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))
  v14.assign(tf.reshape(f14,tf.shape(v14)))
  v15.assign(tf.reshape(f15,tf.shape(v15)))
  v16.assign(tf.reshape(f16,tf.shape(v16)))
  v17.assign(tf.reshape(f17,tf.shape(v17)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))
  v19.assign(tf.reshape(f19,tf.shape(v19)))
  v20.assign(tf.reshape(f20,tf.shape(v20)))
  v21.assign(tf.reshape(f21,tf.shape(v21)))
  v22.assign(tf.reshape(f22,tf.shape(v22)))
  v23.assign(tf.reshape(f23,tf.shape(v23)))
  v24.assign(tf.reshape(f24,tf.shape(v24)))
  v25.assign(tf.reshape(f25,tf.shape(v25)))
  v26.assign(tf.reshape(f26,tf.shape(v26)))
  v27.assign(tf.reshape(f27,tf.shape(v27)))
  v28.assign(tf.reshape(f28,tf.shape(v28)))
  v29.assign(tf.reshape(f29,tf.shape(v29)))
  v30.assign(tf.reshape(f30,tf.shape(v30)))
  v31.assign(tf.reshape(f31,tf.shape(v31)))
  v32.assign(tf.reshape(f32,tf.shape(v32)))
  v33.assign(tf.reshape(f33,tf.shape(v33)))
  v34.assign(tf.reshape(f34,tf.shape(v34)))
  v35.assign(tf.reshape(f35,tf.shape(v35)))
  v36.assign(tf.reshape(f36,tf.shape(v36)))
  v37.assign(tf.reshape(f37,tf.shape(v37)))
  v38.assign(tf.reshape(f38,tf.shape(v38)))
  v39.assign(tf.reshape(f39,tf.shape(v39)))
  v40.assign(tf.reshape(f40,tf.shape(v40)))
  v41.assign(tf.reshape(f41,tf.shape(v41)))
  v42.assign(tf.reshape(f42,tf.shape(v42)))
  v43.assign(tf.reshape(f43,tf.shape(v43)))
  v44.assign(tf.reshape(f44,tf.shape(v44)))
  v45.assign(tf.reshape(f45,tf.shape(v45)))
  v46.assign(tf.reshape(f46,tf.shape(v46)))
  v47.assign(tf.reshape(f47,tf.shape(v47)))
  v48.assign(tf.reshape(f48,tf.shape(v48)))
  v49.assign(tf.reshape(f49,tf.shape(v49)))
  v50.assign(tf.reshape(f50,tf.shape(v50)))
  v51.assign(tf.reshape(f51,tf.shape(v51)))
  v52.assign(tf.reshape(f52,tf.shape(v52)))
  v53.assign(tf.reshape(f53,tf.shape(v53)))
  v54.assign(tf.reshape(f54,tf.shape(v54)))
  v55.assign(tf.reshape(f55,tf.shape(v55)))
  v56.assign(tf.reshape(f56,tf.shape(v56)))
  v57.assign(tf.reshape(f57,tf.shape(v57)))
  v58.assign(tf.reshape(f58,tf.shape(v58)))
  v59.assign(tf.reshape(f59,tf.shape(v59)))
  v60.assign(tf.reshape(f60,tf.shape(v60)))
  v61.assign(tf.reshape(f61,tf.shape(v61)))
  v62.assign(tf.reshape(f62,tf.shape(v62)))
  v63.assign(tf.reshape(f63,tf.shape(v63)))
  v64.assign(tf.reshape(f64,tf.shape(v64)))

def set_flat70(flat,flat_sizes,
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
    v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,
    v31,v32,v33,v34,v35,v36,v37,v38,v39,v40,
    v41,v42,v43,v44,v45,v46,v47,v48,v49,v50,
    v51,v52,v53,v54,v55,v56,v57,v58,v59,v60,
    v61,v62,v63,v64,v65,v66,v67,v68,v69,v70):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,\
  f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,\
  f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,\
  f31,f32,f33,f34,f35,f36,f37,f38,f39,f40,\
  f41,f42,f43,f44,f45,f46,f47,f48,f49,f50,\
  f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,\
  f61,f62,f63,f64,f65,f66,f67,f68,f69,f70= tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))
  v14.assign(tf.reshape(f14,tf.shape(v14)))
  v15.assign(tf.reshape(f15,tf.shape(v15)))
  v16.assign(tf.reshape(f16,tf.shape(v16)))
  v17.assign(tf.reshape(f17,tf.shape(v17)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))
  v19.assign(tf.reshape(f19,tf.shape(v19)))
  v20.assign(tf.reshape(f20,tf.shape(v20)))
  v21.assign(tf.reshape(f21,tf.shape(v21)))
  v22.assign(tf.reshape(f22,tf.shape(v22)))
  v23.assign(tf.reshape(f23,tf.shape(v23)))
  v24.assign(tf.reshape(f24,tf.shape(v24)))
  v25.assign(tf.reshape(f25,tf.shape(v25)))
  v26.assign(tf.reshape(f26,tf.shape(v26)))
  v27.assign(tf.reshape(f27,tf.shape(v27)))
  v28.assign(tf.reshape(f28,tf.shape(v28)))
  v29.assign(tf.reshape(f29,tf.shape(v29)))
  v30.assign(tf.reshape(f30,tf.shape(v30)))
  v31.assign(tf.reshape(f31,tf.shape(v31)))
  v32.assign(tf.reshape(f32,tf.shape(v32)))
  v33.assign(tf.reshape(f33,tf.shape(v33)))
  v34.assign(tf.reshape(f34,tf.shape(v34)))
  v35.assign(tf.reshape(f35,tf.shape(v35)))
  v36.assign(tf.reshape(f36,tf.shape(v36)))
  v37.assign(tf.reshape(f37,tf.shape(v37)))
  v38.assign(tf.reshape(f38,tf.shape(v38)))
  v39.assign(tf.reshape(f39,tf.shape(v39)))
  v40.assign(tf.reshape(f40,tf.shape(v40)))
  v41.assign(tf.reshape(f41,tf.shape(v41)))
  v42.assign(tf.reshape(f42,tf.shape(v42)))
  v43.assign(tf.reshape(f43,tf.shape(v43)))
  v44.assign(tf.reshape(f44,tf.shape(v44)))
  v45.assign(tf.reshape(f45,tf.shape(v45)))
  v46.assign(tf.reshape(f46,tf.shape(v46)))
  v47.assign(tf.reshape(f47,tf.shape(v47)))
  v48.assign(tf.reshape(f48,tf.shape(v48)))
  v49.assign(tf.reshape(f49,tf.shape(v49)))
  v50.assign(tf.reshape(f50,tf.shape(v50)))
  v51.assign(tf.reshape(f51,tf.shape(v51)))
  v52.assign(tf.reshape(f52,tf.shape(v52)))
  v53.assign(tf.reshape(f53,tf.shape(v53)))
  v54.assign(tf.reshape(f54,tf.shape(v54)))
  v55.assign(tf.reshape(f55,tf.shape(v55)))
  v56.assign(tf.reshape(f56,tf.shape(v56)))
  v57.assign(tf.reshape(f57,tf.shape(v57)))
  v58.assign(tf.reshape(f58,tf.shape(v58)))
  v59.assign(tf.reshape(f59,tf.shape(v59)))
  v60.assign(tf.reshape(f60,tf.shape(v60)))
  v61.assign(tf.reshape(f61,tf.shape(v61)))
  v62.assign(tf.reshape(f62,tf.shape(v62)))
  v63.assign(tf.reshape(f63,tf.shape(v63)))
  v64.assign(tf.reshape(f64,tf.shape(v64)))
  v65.assign(tf.reshape(f65,tf.shape(v65)))
  v66.assign(tf.reshape(f66,tf.shape(v66)))
  v67.assign(tf.reshape(f67,tf.shape(v67)))
  v68.assign(tf.reshape(f68,tf.shape(v68)))
  v69.assign(tf.reshape(f69,tf.shape(v69)))
  v70.assign(tf.reshape(f70,tf.shape(v70)))

def set_flat76(flat,flat_sizes,
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
    v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,
    v31,v32,v33,v34,v35,v36,v37,v38,v39,v40,
    v41,v42,v43,v44,v45,v46,v47,v48,v49,v50,
    v51,v52,v53,v54,v55,v56,v57,v58,v59,v60,
    v61,v62,v63,v64,v65,v66,v67,v68,v69,v70,
    v71,v72,v73,v74,v75,v76):
  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,\
  f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,\
  f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,\
  f31,f32,f33,f34,f35,f36,f37,f38,f39,f40,\
  f41,f42,f43,f44,f45,f46,f47,f48,f49,f50,\
  f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,\
  f61,f62,f63,f64,f65,f66,f67,f68,f69,f70,\
  f71,f72,f73,f74,f75,f76 = tf.split(flat,flat_sizes)
  v1.assign(tf.reshape(f1,tf.shape(v1)))
  v2.assign(tf.reshape(f2,tf.shape(v2)))
  v3.assign(tf.reshape(f3,tf.shape(v3)))
  v4.assign(tf.reshape(f4,tf.shape(v4)))
  v5.assign(tf.reshape(f5,tf.shape(v5)))
  v6.assign(tf.reshape(f6,tf.shape(v6)))
  v7.assign(tf.reshape(f7,tf.shape(v7)))
  v8.assign(tf.reshape(f8,tf.shape(v8)))
  v9.assign(tf.reshape(f9,tf.shape(v9)))
  v10.assign(tf.reshape(f10,tf.shape(v10)))
  v11.assign(tf.reshape(f11,tf.shape(v11)))
  v12.assign(tf.reshape(f12,tf.shape(v12)))
  v13.assign(tf.reshape(f13,tf.shape(v13)))
  v14.assign(tf.reshape(f14,tf.shape(v14)))
  v15.assign(tf.reshape(f15,tf.shape(v15)))
  v16.assign(tf.reshape(f16,tf.shape(v16)))
  v17.assign(tf.reshape(f17,tf.shape(v17)))
  v18.assign(tf.reshape(f18,tf.shape(v18)))
  v19.assign(tf.reshape(f19,tf.shape(v19)))
  v20.assign(tf.reshape(f20,tf.shape(v20)))
  v21.assign(tf.reshape(f21,tf.shape(v21)))
  v22.assign(tf.reshape(f22,tf.shape(v22)))
  v23.assign(tf.reshape(f23,tf.shape(v23)))
  v24.assign(tf.reshape(f24,tf.shape(v24)))
  v25.assign(tf.reshape(f25,tf.shape(v25)))
  v26.assign(tf.reshape(f26,tf.shape(v26)))
  v27.assign(tf.reshape(f27,tf.shape(v27)))
  v28.assign(tf.reshape(f28,tf.shape(v28)))
  v29.assign(tf.reshape(f29,tf.shape(v29)))
  v30.assign(tf.reshape(f30,tf.shape(v30)))
  v31.assign(tf.reshape(f31,tf.shape(v31)))
  v32.assign(tf.reshape(f32,tf.shape(v32)))
  v33.assign(tf.reshape(f33,tf.shape(v33)))
  v34.assign(tf.reshape(f34,tf.shape(v34)))
  v35.assign(tf.reshape(f35,tf.shape(v35)))
  v36.assign(tf.reshape(f36,tf.shape(v36)))
  v37.assign(tf.reshape(f37,tf.shape(v37)))
  v38.assign(tf.reshape(f38,tf.shape(v38)))
  v39.assign(tf.reshape(f39,tf.shape(v39)))
  v40.assign(tf.reshape(f40,tf.shape(v40)))
  v41.assign(tf.reshape(f41,tf.shape(v41)))
  v42.assign(tf.reshape(f42,tf.shape(v42)))
  v43.assign(tf.reshape(f43,tf.shape(v43)))
  v44.assign(tf.reshape(f44,tf.shape(v44)))
  v45.assign(tf.reshape(f45,tf.shape(v45)))
  v46.assign(tf.reshape(f46,tf.shape(v46)))
  v47.assign(tf.reshape(f47,tf.shape(v47)))
  v48.assign(tf.reshape(f48,tf.shape(v48)))
  v49.assign(tf.reshape(f49,tf.shape(v49)))
  v50.assign(tf.reshape(f50,tf.shape(v50)))
  v51.assign(tf.reshape(f51,tf.shape(v51)))
  v52.assign(tf.reshape(f52,tf.shape(v52)))
  v53.assign(tf.reshape(f53,tf.shape(v53)))
  v54.assign(tf.reshape(f54,tf.shape(v54)))
  v55.assign(tf.reshape(f55,tf.shape(v55)))
  v56.assign(tf.reshape(f56,tf.shape(v56)))
  v57.assign(tf.reshape(f57,tf.shape(v57)))
  v58.assign(tf.reshape(f58,tf.shape(v58)))
  v59.assign(tf.reshape(f59,tf.shape(v59)))
  v60.assign(tf.reshape(f60,tf.shape(v60)))
  v61.assign(tf.reshape(f61,tf.shape(v61)))
  v62.assign(tf.reshape(f62,tf.shape(v62)))
  v63.assign(tf.reshape(f63,tf.shape(v63)))
  v64.assign(tf.reshape(f64,tf.shape(v64)))
  v65.assign(tf.reshape(f65,tf.shape(v65)))
  v66.assign(tf.reshape(f66,tf.shape(v66)))
  v67.assign(tf.reshape(f67,tf.shape(v67)))
  v68.assign(tf.reshape(f68,tf.shape(v68)))
  v69.assign(tf.reshape(f69,tf.shape(v69)))
  v70.assign(tf.reshape(f70,tf.shape(v70)))
  v71.assign(tf.reshape(f71,tf.shape(v71)))
  v72.assign(tf.reshape(f72,tf.shape(v72)))
  v73.assign(tf.reshape(f73,tf.shape(v73)))
  v74.assign(tf.reshape(f74,tf.shape(v74)))
  v75.assign(tf.reshape(f75,tf.shape(v75)))
  v76.assign(tf.reshape(f76,tf.shape(v76)))

set_flat_call = {
  1: tf.function(set_flat1),
  2: tf.function(set_flat2),
  6: tf.function(set_flat6),
  7: tf.function(set_flat7),
  8: tf.function(set_flat8),
  10: tf.function(set_flat10),
  12: tf.function(set_flat12),
  13: tf.function(set_flat13),
  18: tf.function(set_flat18),
  19: tf.function(set_flat19),
  20: tf.function(set_flat20),
  24: tf.function(set_flat24),
  25: tf.function(set_flat25),
  30: tf.function(set_flat30),
  31: tf.function(set_flat31),
  34: tf.function(set_flat34),
  35: tf.function(set_flat35),
  36: tf.function(set_flat36),
  37: tf.function(set_flat37),
  42: tf.function(set_flat42),
  48: tf.function(set_flat48),
  58: tf.function(set_flat58),
  64: tf.function(set_flat64),
  70: tf.function(set_flat70),
  76: tf.function(set_flat76)
}

class SetFlat(object):
  def __init__(self, var_list):
    self.var_list = var_list
    self.shapes = [var.shape.as_list() for var in var_list]
    self.flat_sizes = tf.constant([int(np.prod(shape)) for shape in self.shapes], tf.int32)
    self.fn = set_flat_call[len(var_list)]
  def __call__(self, θ):
    self.fn(θ, self.flat_sizes, *self.var_list)

class GetFlat(object):
  def __init__(self, var_list):
    self.var_list = var_list
  @tf.function
  def __call__(self):
    return tf.concat([tf.reshape(v, [-1]) for v in self.var_list], 0)

class Optimizer:
  def __init__(self, var_list):
    self.var_list = var_list
    self.set_flat = SetFlat(var_list)
    self.get_flat = GetFlat(var_list)
    self.buffer = np.empty(int(tf.reduce_sum(self.set_flat.flat_sizes)), dtype=np.float32)
  def update(self, localg, stepsize):
    raise NotImplementedError

class SGD(Optimizer):
  def __init__(self, var_list):
    super().__init__(var_list)
    total_size = sum([int(np.prod(v.shape.as_list())) for v in self.var_list])

  @tf.function
  def update_tf(self, globalg, stepsize):
    θ_before = self.get_flat()
    θ = θ_before - stepsize * globalg
    θ = tf.where(tf.math.is_finite(θ), θ, θ_before)
    self.set_flat(θ)

  def update(self, localg, stepsize):
    self.update_tf(localg, stepsize)

class Adam(Optimizer):
  def __init__(self, var_list, *, β1=0.9, β2=0.999):
    self.β1 = tf.constant(β1, tf.float32)
    self.β2 = tf.constant(β2, tf.float32)
    super().__init__(var_list)
    total_size = sum([int(np.prod(v.shape.as_list())) for v in self.var_list])

    self.m = tf.Variable(tf.zeros(total_size, tf.float32))
    self.v = tf.Variable(tf.zeros(total_size, tf.float32))
    self.t = tf.Variable(0, dtype=tf.int32)

  @tf.function
  def reset(self):
    self.m.assign(0.*self.m) 
    self.v.assign(0.*self.v)
    self.t.assign(0)

  @tf.function
  def update_tf(self, globalg, stepsize):
    β1 = self.β1
    β2 = self.β2
    self.t.assign_add(1)
    t = tf.cast(self.t, tf.float32)
    a = stepsize * tf.sqrt(1 - tf.math.pow(β2, t))/(1 - tf.math.pow(β1,t))
    self.m.assign(β1 * self.m + (1 - β1) * globalg)
    self.v.assign(β2 * self.v + (1 - β2) * tf.square(globalg))
    step = - a * self.m / (tf.sqrt(self.v) + 1e-8)
    θ_before = self.get_flat()
    θ = θ_before + step
    θ = tf.where(tf.math.is_finite(θ), θ, θ_before)
    self.set_flat(θ)

  def update(self, localg, stepsize):
    self.update_tf(localg, stepsize)

class AdaBelief(Optimizer):
  def __init__(self, var_list, *, β1=0.9, β2=0.999):
    self.β1 = tf.constant(β1, tf.float32)
    self.β2 = tf.constant(β2, tf.float32)
    super().__init__(var_list)
    total_size = sum([int(np.prod(v.shape.as_list())) for v in self.var_list])

    self.m = tf.Variable(tf.zeros(total_size, tf.float32))
    self.s = tf.Variable(tf.zeros(total_size, tf.float32))
    self.t = tf.Variable(0, dtype=tf.int32)

  @tf.function
  def reset(self):
    self.m.assign(0.*self.m)
    self.s.assign(0.*self.s)
    self.t.assign(0)

  @tf.function
  def update_tf(self, globalg, stepsize):
    β1 = self.β1
    β2 = self.β2
    self.t.assign_add(1)
    t = tf.cast(self.t, tf.float32)
    a = stepsize * tf.sqrt(1 - tf.math.pow(β2, t))/(1 - tf.math.pow(β1,t))
    self.m.assign(β1 * self.m + (1 - β1) * globalg)
    self.s.assign(β2 * self.s + (1 - β2) * tf.square(globalg-self.m) + 1e-8)
    step = - a * self.m / (tf.sqrt(self.s) + 1e-8)
    θ_before = self.get_flat()
    θ = θ_before + step
    θ = tf.where(tf.math.is_finite(θ), θ, θ_before)
    self.set_flat(θ)

  def update(self, localg, stepsize):
    self.update_tf(localg, stepsize)

