from math import *

def get_area(f, a, b, flag = "left"):
    if flag == "left":
        return f(a)*(b - a)
    elif flag == "right":
        return f(b)*(b - a)
    else:
        raise "Unresolved flag. Use \"left\" or \"right\" "

def rectangle_method(f, a, b, flag = "left", n = 1):
    sum = 0
    for i in range(n):
        sum += get_area(f, a+(b-a)/n*i, a+(b-a)/n*(i+1), flag)
    return sum
