def get_area(f, a, b):
    return f((a+b)/2)*abs(b - a)

def mean_method(f, a, b, n = 1):
    sum = 0
    for i in range(n):
        sum += get_area(f, a+(b-a)/n*i, a+(b-a)/n*(i+1))
    return sum