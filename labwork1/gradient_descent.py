def gradient_descent(derivative_func, initial_x, learning_rate, num_iterations):
    x = initial_x
    for i in range(num_iterations):
        grad = derivative_func(x)
        x_new = x - learning_rate * grad
        print(f"Iteration {i+1}: x = {x_new}, f(x) = {x_new**2}")
        x = x_new

def f(x):
    return x**2

def derivative(x):
    return 2*x

initial_x = 10
learning_rate = 0.1
num_iterations = 50

gradient_descent(derivative, initial_x, learning_rate, num_iterations)
