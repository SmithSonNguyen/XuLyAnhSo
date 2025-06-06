import numpy as np

m = 3
n = 3
sigma = 1.0
w = np.zeros((m,n), np.float64)
a = m // 2
b = n // 2
for s in range(-a, a+1):
    for t in range(-b, b+1):
        r = np.exp(-(s*s + t*t)/(2*sigma*sigma))
        w[s+a,t+b] = r

for i in range(0, m):
    for j in range(0, n):
        print('%8.4f' % w[i,j], end = '')
    print()

K = np.sum(w)
print('K = %.4f' % K)
w = w/K
for i in range(0, m):
    for j in range(0, n):
        print('%8.4f' % w[i,j], end = '')
    print()
K = np.sum(w)
print('K = %.4f' % K)
