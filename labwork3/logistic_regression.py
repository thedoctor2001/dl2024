import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def compute_loss(y, y_pred):
    return -sum(yi * math.log(y_pred_i) + (1 - yi) * math.log(1 - y_pred_i) for yi, y_pred_i in zip(y, y_pred)) / len(y)

def compute_gradients(X, y, y_pred):
    gradients = [0] * len(X[0])
    for i in range(len(X[0])): 
        gradients[i] = sum((y_pred[j] - y[j]) * X[j][i] for j in range(len(X))) / len(X)
    return gradients

def gradient_descent(X, y, learning_rate=0.01, iterations=100):
    weights = [0] * len(X[0])

    for iteration in range(iterations):
        z = [sum(weights[i] * X[j][i] for i in range(len(X[0]))) for j in range(len(X))]
        y_pred = [sigmoid(zi) for zi in z]
        loss = compute_loss(y, y_pred)
        gradients = compute_gradients(X, y, y_pred)
        weights = [weights[i] - learning_rate * gradients[i] for i in range(len(weights))]

        print(f"Iteration {iteration + 1}: Loss = {loss:.4f}, Weights = {weights}")
    
    return weights

def read_data(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip().split(',') for line in file if line.strip()]
    return data

def main():
    data_path = 'loan_data.csv'
    raw_data = read_data(data_path)

    X = [[1] + [float(x) for x in row[:-1]] for row in raw_data] 
    y = [int(row[-1]) for row in raw_data]

    final_weights = gradient_descent(X, y, learning_rate=0.01, iterations=100)
    print("Final model weights:", final_weights)

if __name__ == "__main__":
    main()