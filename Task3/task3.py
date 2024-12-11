from cProfile import label
from math import *
import numpy as np
import rectange_method
from Task3.Simpson import simpson_method
from Task3.mean import mean_method
from Task3.trapezoid import trapezoid_method
from Task3.rectange_method import rectangle_method
from matplotlib import pyplot as plt

def f(x):
    return 1/(1+x**2)

def f2(t):
    return exp(-t**2)

def erf(x):
    return 2/np.sqrt(pi)*simpson_method(f2, 0, x, 10000)

n = []
rect_res = []
trap_res = []
mean_res = []
simp_res = []

for i in range(20):
    poww = 2
    n.append(poww**i)
    rect = rectangle_method(f, -1, 1, "left", poww**i)
    rect_res.append(abs(pi/2-rect))
    trap = trapezoid_method(f, -1, 1, poww**i)
    trap_res.append(abs(pi/2-trap))
    mean_res.append(abs(pi/2-mean_method     (f, -1, 1, poww**i)))
    simp_res.append(abs(pi/2-simpson_method  (f, -1, 1, poww**i)))


plt.plot(n, rect_res, label = "Rectangle", color = "blue")
plt.plot(n, trap_res, label = "Trapezoid", color = "red")
plt.plot(n, mean_res, label = "Mean", color = "green")
plt.plot(n, simp_res, label = "Simpson", color = "black")

plt.xscale('log')
plt.yscale('log')

plt.legend()
plt.show()

#print(erf(float(input("Enter x: "))))

