import matplotlib.pyplot as plt

a = 10
b = 2
c = 2
d = 10

def f1(x, y):
    return a * x - b * x * y


def f2(x, y):
    return c * x * y - d * y

def runge_kutta_2nd_order(x0_, y0_, t0_, h_, N_):
    x_values_ = [x0_]
    y_values_ = [y0_]
    t_values_ = [t0_]

    x = x0_
    y = y0_
    t = t0_

    for i in range(N_):
        k1 = f1(x, y)
        l1 = f2(x, y)

        k2 = f1(x + h_ * k1 / 2, y + h_ * l1 / 2)
        l2 = f2(x + h_ * k1 / 2, y + h_ * l1 / 2)

        x = x + h_ * k2
        y = y + h_ * l2
        t = t + h_

        x_values_.append(x)
        y_values_.append(y)
        t_values_.append(t)

    return x_values_, y_values_, t_values_

x0 = 1
y0 = 1
t0 = 0
h = 0.001
N = 10000

x_values, y_values, t_values = runge_kutta_2nd_order(x0, y0, t0, h, N)

plt.plot(x_values, y_values)

#plt.plot(t_values, x_values)
#plt.plot(t_values, y_values)
plt.xlabel('x (Численность жертвы)')
plt.ylabel('y (Численность хищника)')
plt.title('Фазовая траектория системы хищник-жертва')
plt.grid(True)
plt.show()
