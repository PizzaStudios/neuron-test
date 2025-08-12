/**
 * Generates a random number from a normal (Gaussian) distribution using Box-Muller transform.
 * @param {number} [mean=0] - Mean of the distribution.
 * @param {number} [stddev=1] - Standard deviation of the distribution.
 * @returns {number} Random number from the normal distribution.
 */
function randomNormal(mean = 0, stddev = 1) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z * stddev + mean;
}

/**
 * Class representing a single neuron with configurable activation and training.
 */
class Neuron {
    /**
     * Creates a new Neuron instance.
     * @param {number} inputs - Number of inputs (features).
     * @param {string} [activationType='relu'] - Activation function type: 'relu', 'leakyrelu', 'sigmoid', 'tanh', or 'linear'.
     * @param {number} [learningRate=0.01] - Learning rate used for weight updates.
     * @throws {Error} Throws error if input parameters are invalid.
     */
    constructor(inputs, activationType = 'relu', learningRate = 0.01) {
        if (!Number.isInteger(inputs)) {
            throw new Error("inputs must be an integer");
        }
        this.inputs = inputs;

        if (typeof activationType !== 'string') {
            throw new Error("activationType must be a string");
        }
        this.activationType = activationType;

        if (!Number.isFinite(learningRate)) {
            throw new Error("learningRate must be a number");
        }
        this.learningRate = learningRate;

        if (this.activationType === 'relu' || this.activationType === 'leakyrelu') {
            // He initialization for ReLU and LeakyReLU
            this.weights = Array.from({ length: this.inputs }, () => randomNormal(0, Math.sqrt(2 / this.inputs)));
            this.bias = 0.01;
        } else {
            // Xavier initialization for other activations
            const limit = Math.sqrt(6 / (this.inputs + 1));
            this.weights = Array.from({ length: this.inputs }, () => Math.random() * 2 * limit - limit);
            this.bias = 0;
        }
    }

    /**
     * Activation function applied to input value.
     * @param {number} x - Weighted sum input.
     * @returns {number} Activated output.
     * @throws {Error} Throws if activation type is unsupported.
     */
    activate(x) {
        switch (this.activationType) {
            case 'relu':
            case 'leakyrelu':
                return x > 0 ? x : 0.01 * x;
            case 'sigmoid':
                return 1 / (1 + Math.exp(-x));
            case 'tanh':
                return Math.tanh(x);
            case 'linear':
                return x;
            default:
                throw new Error("Unsupported activation type");
        }
    }

    /**
     * Derivative of the activation function with respect to input x.
     * Used for backpropagation gradient calculation.
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
            case 'relu':
                return x > 0 ? 1 : 0;
            case 'leakyrelu':
                return x > 0 ? 1 : 0.01;
            case 'tanh': {
                const t = Math.tanh(x);
                return 1 - t * t;
            }
            case 'linear':
                return 1;
            default:
                throw new Error("Unsupported activation type");
        }
    }

    /**
     * Predicts output of the neuron for given inputs.
     * @param {number[]} x - Input feature array.
     * @returns {{z: number, output: number}} Object containing weighted sum `z` and activated output.
     * @throws {Error} Throws if input length does not match weights length.
     */
    predict(x) {
        if (!Array.isArray(x)) throw new Error("Input must be an array");
        if (x.length !== this.weights.length) {
            throw new Error("Input size does not match number of weights.");
        }
        const z = this.weights.reduce((sum, w, i) => sum + w * x[i], this.bias);
        return { z, output: this.activate(z) };
    }

    /**
     * Trains the neuron on a single sample using gradient descent.
     * @param {number[]} x - Input features.
     * @param {number} y - Target output.
     * @throws {Error} Throws if x is not array or size mismatch.
     */
    train(x, y) {
        if (!Array.isArray(x)) {
            throw new Error('x must be an array');
        }
        const { z, output } = this.predict(x);
        const error = y - output;
        const gradient = error * this.derivative(z);

        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] += this.learningRate * gradient * x[i];
        }
        this.bias += this.learningRate * gradient;
    }

    /**
     * Trains the neuron on a dataset for multiple epochs.
     * @param {number[][]} x - Array of input feature arrays.
     * @param {number[]} y - Array of target outputs.
     * @param {number} [epochs=50] - Number of training epochs.
     * @throws {Error} Throws if input arrays length mismatch or epochs is not a number.
     */
    trainset(x, y, epochs = 50) {
        if (!Array.isArray(x) || !Array.isArray(y)) {
            throw new Error('x and y must be arrays');
        }
        if (x.length !== y.length) {
            throw new Error('X and Y lengths are not matching');
        }
        if (typeof epochs !== 'number') {
            throw new Error('epochs must be a number');
        }

        for (let epoch = 0; epoch < epochs; epoch++) {
            for (let i = 0; i < x.length; i++) {
                this.train(x[i], y[i]);
            }
        }
    }
}
module.exports = {Neuron}