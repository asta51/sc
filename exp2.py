import numpy as np

# Define the Discrete Perceptron class
class DiscretePerceptron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0

    def predict(self, inputs):
        activation = np.dot(self.weights, inputs) + self.bias
        return 1 if activation > 0 else 0  # Step activation function

    def train(self, inputs, targets, learning_rate=0.1, epochs=100):
        for _ in range(epochs):
            for x, y in zip(inputs, targets):
                prediction = self.predict(x)
                error = y - prediction
                self.weights += learning_rate * error * x
                self.bias += learning_rate * error


# Main function to run the perceptron
def main():
    # Training data
    class_0 = np.array([[2, 3], [3, 2], [1, 1]])  # Class 0
    class_1 = np.array([[5, 7], [6, 8], [7, 6]])  # Class 1

    # Combine data and assign labels
    inputs = np.vstack((class_0, class_1))  # Stack vertically
    targets = np.array([0, 0, 0, 1, 1, 1])  # Labels

    # Initialize perceptron
    perceptron = DiscretePerceptron(input_size=2)

    # Train perceptron
    perceptron.train(inputs, targets)

    # Test with new data
    test_data = np.array([[4, 5], [2, 2]])
    for data in test_data:
        prediction = perceptron.predict(data)
        if prediction == 0:
            print(f"Data {data} belongs to class 0")
        else:
            print(f"Data {data} belongs to class 1")


# Run the main function
if __name__ == "__main__":
    main()
