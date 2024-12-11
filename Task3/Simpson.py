def get_area(f, a, b):
    return (b-a)/6*(f(a)+4*f((a+b)/2)+f(b))

def simpson_method(f, a, b, n = 1):
    sum = 0
    for i in range(n):
        sum += get_area(f, a+(b-a)/n*i, a+(b-a)/n*(i+1))
    return sum