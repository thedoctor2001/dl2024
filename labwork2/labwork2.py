def load_data(csv_file):
    x = []
    y = []
    with open(csv_file, 'r') as file:
        next(file)
        for line in file:
            line = line.strip().split(',')
            x.append(float(line[0]))
            y.append(float(line[1]))
    return x, y

def compute_cost(x, y, w1, w0):
    m = len(y)
    total_cost = 0
    for i in range(m):
        y_pred = w1 * x[i] + w0
        total_cost += (y_pred - y[i]) ** 2
    return total_cost / (2 * m)

def gradient_descent(x, y, w1, w0, learning_rate, iterations, tolerance=1e-6):
    m = len(y)
    previous_cost = compute_cost(x, y, w1, w0)
    for i in range(iterations):
        sum_errors_w1 = 0
        sum_errors_w0 = 0
        for j in range(m):
            y_pred = w1 * x[j] + w0
            error = y_pred - y[j]
            sum_errors_w1 += error * x[j]
            sum_errors_w0 += error

        w1 -= (learning_rate / m) * sum_errors_w1
        w0 -= (learning_rate / m) * sum_errors_w0

        if i % 10 == 9 or i == iterations - 1:
            current_cost = compute_cost(x, y, w1, w0)
            print(f"Iteration {i}: w1 = {w1}, w0 = {w0}, cost = {current_cost}")

            if abs(current_cost - previous_cost) < tolerance:
                print("Convergence reached.")
                break
            previous_cost = current_cost

    return w1, w0


def linear_regression(csv_file):
    x, y = load_data(csv_file)
    w1, w0 = 0, 0
    learning_rate = 0.001
    iterations = 10000
    w1, w0 = gradient_descent(x, y, w1, w0, learning_rate, iterations)
    return w1, w0

csv_file = 'data.csv'
w1, w0 = linear_regression(csv_file)
print(f"Final results: w1 = {w1}, w0 = {w0}")
