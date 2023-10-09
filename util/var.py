import tensorflow as tf

@tf.function
def var_sync6(s1,s2,s3,s4,s5,s6,v1,v2,v3,v4,v5,v6):
  v1.assign(s1)
  v2.assign(s2)
  v3.assign(s3)
  v4.assign(s4)
  v5.assign(s5)
  v6.assign(s6)

@tf.function
def var_sync12(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,
               v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12):
  v1.assign(s1)
  v2.assign(s2)
  v3.assign(s3)
  v4.assign(s4)
  v5.assign(s5)
  v6.assign(s6)
  v7.assign(s7)
  v8.assign(s8)
  v9.assign(s9)
  v10.assign(s10)
  v11.assign(s11)
  v12.assign(s12)

@tf.function
def var_sync18(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,
               s11,s12,s13,s14,s15,s16,s17,s18,
               v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
               v11,v12,v13,v14,v15,v16,v17,v18):
  v1.assign(s1)
  v2.assign(s2)
  v3.assign(s3)
  v4.assign(s4)
  v5.assign(s5)
  v6.assign(s6)
  v7.assign(s7)
  v8.assign(s8)
  v9.assign(s9)
  v10.assign(s10)
  v11.assign(s11)
  v12.assign(s12)
  v13.assign(s13)
  v14.assign(s14)
  v15.assign(s15)
  v16.assign(s16)
  v17.assign(s17)
  v18.assign(s18)
