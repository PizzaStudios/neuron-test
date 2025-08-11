function randomNormal(mean = 0, stddev = 1) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z * stddev + mean;
}

/**
 * Represents a single neuron with configurable activation and training.
 */
class Neuron {
    /**
     * Creates a new Neuron instance.
     * @param {number} numOfInputs - Number of inputs.
     * @param {string} [activationType='relu'] - Activation function type ('sigmoid', 'relu', 'leakyrelu', 'tanh', 'linear').
     * @param {number} [learningRate=0.1] - Learning rate for weight updates.
     */
    constructor(numOfInputs, activationType = 'relu', learningRate = 0.1) {
        this.numOfInputs = numOfInputs;
        this.activationType = activationType;
        this.learningRate = learningRate;

        if (['relu', 'leakyrelu'].includes(activationType)) {
            const std = Math.sqrt(2 / numOfInputs); // He initialization
            this.weights = Array.from({ length: numOfInputs }, () => randomNormal(0, std));
        } else {
            const limit = Math.sqrt(6 / (numOfInputs + 1)); // Xavier initialization
            this.weights = Array.from({ length: numOfInputs }, () => Math.random() * 2 * limit - limit);
        }

        this.bias = Math.random() * 2 - 1;
        this.lastInputs = [];
        this.lastWeightedSum = 0;
        this.lastOutput = 0;
    }

    /**
     * Activation function applied to input value.
     * @param {number} x - Weighted sum input.
     * @returns {number} Activated output.
     * @throws {Error} Throws if activation type is unsupported.
     */
    activate(x) {
        switch (this.activationType) {
            case 'sigmoid': return 1 / (1 + Math.exp(-x));
            case 'relu': return Math.max(0, x);
            case 'leakyrelu': return x > 0 ? x : 0.01 * x;
            case 'tanh': return Math.tanh(x);
            case 'linear': return x;
            default: throw new Error("Unsupported activation type");
        }
    }

    /**
     * Derivative of the activation function with respect to input x.
     * @param {number} x - Weighted sum input.
     * @returns {number} Derivative value.
     * @throws {Error} Throws if activation type is unsupported.
     */
    derivative(x) {
        switch (this.activationType) {
            case 'sigmoid': {
                const s = 1 / (1 + Math.exp(-x));
                return s * (1 - s);
            }
            case 'relu': return x > 0 ? 1 : 0;
            case 'leakyrelu': return x > 0 ? 1 : 0.01;
            case 'tanh': {
                const t = Math.tanh(x);
                return 1 - t * t;
            }
            case 'linear': return 1;
            default: throw new Error("Unsupported activation type");
        }
    }

    /**
     * Feeds input through the neuron, computing weighted sum plus bias and applying activation.
     * @param {number[]} inputs - Array of input values.
     * @returns {number} The output after activation.
     * @throws {Error} Throws if input array length doesn't match number of weights.
     */
    feed(inputs) {
        if (!Array.isArray(inputs)) throw new Error('Inputs must be an array');
        if (inputs.length !== this.numOfInputs) throw new Error('Input length mismatch');
        this.lastInputs = inputs;
        this.lastWeightedSum = this.weights.reduce((sum, w, i) => sum + w * inputs[i], 0) + this.bias;
        this.lastOutput = this.activate(this.lastWeightedSum);
        return this.lastOutput;
    }

    /**
     * Updates weights and bias based on delta error and stored inputs.
     * @param {number} delta - Error gradient for this neuron.
     */
    train(delta) {
        for (let i = 0; i < this.numOfInputs; i++) {
            this.weights[i] += this.learningRate * delta * this.lastInputs[i];
        }
        this.bias += this.learningRate * delta;
    }
}



/**
 * Represents a simple feedforward neural network with softmax output.
 */
class SoftmaxNN {
    /**
     * Creates a new Softmax Neural Network.
     * @param {number} inputs - Number of input features.
     * @param {number[]} hidden - Array specifying number of neurons in each hidden layer.
     * @param {number} output - Number of output neurons/classes.
     * @param {number} [alpha=0.01] - Optional parameter, learning rate modifier (usage depends on implementation).
     * @param {number} [learningRate=0.1] - Learning rate for neuron training.
     * @throws {Error} Throws if input parameters are invalid.
     */
    constructor(inputs, hidden, output, alpha = 0.01, learningRate = 0.1) {
        if (!inputs) {
            throw new Error('Please input a number for "input". It is recommended to do so.');
        } else {
            this.inputs = inputs;
        }
        if (hidden && Array.isArray(hidden)) {
            this.hidden = hidden;
        } else {
            throw new Error('"hidden" must be an Array');
        }
        if (!output) {
            throw new Error('Please input a number for "output".');
        } else {
            this.output = output;
        }
        this.alpha = alpha;
        let prevSize = inputs;
        this.layers = [];
        for (let size of hidden) {
            this.layers.push(
                Array.from({ length: size }, () => new Neuron(prevSize, 'leakyrelu', learningRate))
            );
            prevSize = size;
        }
        this.outputLayer = Array.from({ length: output }, () => new Neuron(prevSize, 'linear', learningRate));
    }

    /**
     * Applies the softmax function to convert logits into probabilities.
     * @param {number[]} logits - Array of raw output scores.
     * @returns {number[]} Normalized probabilities summing to 1.
     */
    softmax(logits) {
        const max = Math.max(...logits);
        const exps = logits.map(x => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => e / sum);
    }

    /**
     * Performs a forward pass on the network to get predicted probabilities.
     * @param {number[]} input - Input feature vector.
     * @returns {number[]} Output probabilities after softmax.
     * @throws {Error} Throws if input is invalid or has incorrect length.
     */
    predict(input) {
        if (!Array.isArray(input)) {
            throw new Error('Input must be an array of numbers');
        }
        if (input.length !== this.inputs) {
            throw new Error(`Input length mismatch: expected ${this.inputs}, got ${input.length}`);
        }

        let activations = input;
        for (const layer of this.layers) {
            activations = layer.map(neuron => neuron.feed(activations));
        }
        const logits = this.outputLayer.map(neuron => neuron.feed(activations));
        return this.softmax(logits);
    }
}

// === Helpers ===

function shuffleArray(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
}

function oneHotEncode(label, size = 10) {
    const arr = Array(size).fill(0);
    arr[label] = 1;
    return arr;
}

function parseMNISTCsvLine(line) {
    const parts = line.split(',').map(Number);
    const label = parts[0];
    const pixels = parts.slice(1).map(p => p / 255); // normalize to [0, 1]
    return { label, inputs: pixels, expectedOutput: oneHotEncode(label) };
}

module.exports = {
    Neuron,
    SoftmaxNN,
    shuffleArray,
    oneHotEncode,
    parseMNISTCsvLine,

};
