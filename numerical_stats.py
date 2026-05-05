import numpy as np

a = np.float32(1e8) 
b = np.float32(-1e8) 
c = np.float32(1)

print((a + b) + c)  #1
print(a + (b + c)) #0

x = 0.1+0.2
y = 0.3
print(x == y) #False
print(x, y) #0.30000000000000004 0.3

x = np.exp(1000)
y = np.exp(-1000)
print(x) #inf
print(y) #0.0

x = 1e8
y = np.sqrt(x + 1) - np.sqrt(x)
print(y) 

print(type(x))           # <class 'float'>  ← Python 原生 float 就是 float64
print(np.array(x).dtype) # float64