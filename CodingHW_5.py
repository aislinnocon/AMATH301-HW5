
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

########Problem_1############

f = lambda x: x ** 2
fprime = lambda x: 2*x
fdprime  = lambda x: 2
tolerance = 1e-8
x0 = 2
X = np.zeros(101)
X[0] = x0
for k in range(100):
    X[k + 1] = X[k] - fprime(X[k]) / fdprime(X[k])
    if np.abs(fprime(X[k+1])) < tolerance:
        break
X = X[(k+2)]
A1 = k + 2
print("A1 = ", A1)
A2 = X
print("A2 = ", A2)


f = lambda x: x ** 500
fprime = lambda x: 500 * x ** 499
fdprime  = lambda x: 499 * 500 * x ** 498
tolerance = 1e-8
x0 = 2.0
X2 = np.zeros(1001)
X2[0] = x0
for k in range(1000):
    X2[k + 1] = X2[k] - fprime(X2[k]) / fdprime(X2[k])
    if np.abs(fprime(X2[k+1])) < tolerance:
        break
A4 = X2[(k+1)]
A3 = k + 2
print("A3 = ", A3)
print("A4 = ", A4)

f = lambda x: x ** 1000
fprime = lambda x: 1000 * x ** 999
fdprime  = lambda x: 999 * 1000 * x ** 998
tolerance = 1e-8
x0 = 2
X3 = np.ones(1001)
X3[0] = x0
for k in range(1000):
    X3[k + 1] = X3[k] - fprime(X3[k]) / fdprime(X3[k])
    if np.abs(fprime(X3[k+1])) < tolerance:
        break
A6 = X3[(k+1)]
A5 = k + 2
print("A5 = ", A5)
print("A6 = ", A6)


### Part D #####
f = lambda x: x**1000
a = -2
b = 2
c = (-1 + np.sqrt(5)) / 2
x = c * a + (1-c) * b
fx = f(x)
y = (1 - c) * a + c * b
fy = f(y)
n = 0
for k in range(2, 1000):
    if fx < fy:
        b = y
        y = x
        x = c * a + (1-c) * b
        fy = fx
        fx = f(x)
        n = n + 1
    else:
        a = x
        x = y
        y = (1 - c) * a + c * b
        fx = fy
        fy = f(y)
        n = n + 1
    if (b-a) < tolerance:
        break
print("solution =", x)
A8 = x
print(k)
A7 = k + 1
print("A7 = ", A7)


#########Problem_2################
f = lambda t: 1.3*(np.e**(-t/11) - np.e**((-4*t)/3))
a = 1
b = 3
xmin = scipy.optimize.minimize_scalar(lambda v: -f(v), bounds=(a,b), method='Bounded')
print("solution", xmin.x)
A9 = xmin.x
A10 = f(xmin.x)
print(A10)
xvalues = np.linspace(10, 0.01)
plt.plot(xvalues, f(xvalues))
# plt.show()

##########Problem_3############
f = lambda x: -(1/((x-0.3)**2 + 0.01)) - (1/((x-0.9)**2 + 0.04)) - 6
xmin = scipy.optimize.minimize_scalar(f, bounds=(0, 0.5), method='Bounded')
A11 = xmin.x
print("A11 solution", A11)
xmin2 = scipy.optimize.minimize_scalar(lambda v:-f(v), bounds=(0.5, 0.8), method='Bounded')
A12 = xmin2.x
print("A12 solution", A12)
xmin3 = scipy.optimize.minimize_scalar(f, bounds=(0.8, 2), method='Bounded')
A13 = xmin3.x
print("A13 solution", A13)


##########Problem_4############
f = lambda v: (v[0] ** 2 + v[1] - 11) **2 + (v[0] + v[1] ** 2 - 7) ** 2
xmin = scipy.optimize.minimize(f, np.array([-2,3]), method= 'Nelder-Mead')
A14 = xmin.x
A14 = A14.reshape(2,1)
xmax =scipy.optimize.minimize(lambda v:-f(v), np.array([0,0]), method ='Nelder-Mead')
A15 = xmax.x
A15 = A15.reshape(2,1)







