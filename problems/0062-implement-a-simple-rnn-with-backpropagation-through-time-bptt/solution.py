import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        h = np.zeros((self.hidden_size, 1))  
        outputs = []
        self.last_inputs = []
        self.last_hiddens = [h]
        
        for t in range(len(x)):
            self.last_inputs.append(x[t].reshape(-1, 1))
            h = np.tanh(np.dot(self.W_xh, self.last_inputs[t]) + np.dot(self.W_hh, h) + self.b_h)
            y = np.dot(self.W_hy, h) + self.b_y
            outputs.append(y)
            self.last_hiddens.append(h)
        
        self.last_outputs = outputs
        return np.array(outputs)

    def backwar