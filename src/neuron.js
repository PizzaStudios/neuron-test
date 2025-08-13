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
function crossEntropyLoss(pred, target) {
    return -target.reduce((sum, t, i) => sum + t * Math.log(pred[i] + 1e-15), 0);
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
        this.lastInput = null;
        this.z = null;
        this.lastOutput = null;
    
        // For storing delta during backprop
        this.delta = 0;
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
       this.lastInput = x;
       this.z = this.weights.reduce((sum, weight, i) => sum + weight * x[i], this.bias);
       

       this.lastOutput = this.activate(this.z);

       return this.lastOutput
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
        const output = this.predict(x);
        const error = output - y;
        const gradient = error * this.derivative(this.z);

        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] -= this.learningRate * gradient * x[i];
        }
        this.bias -= this.learningRate * gradient;
    }

    backprop({target = null, downstreamDeltaSum = null, prevLayerActivations, prob = null}) {
        if(target !== null) {
            this.delta = (prob ?? this.lastOutput) - target;

        } else {
            this.delta = downstreamDeltaSum * this.derivative(this.z);
            
        }
       
        for(let i = 0; i < this.weights.length; i++) {
            this.weights[i] -= this.learningRate * this.delta * prevLayerActivations[i];
        }
        this.bias -= this.learningRate * this.delta;
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

    /**
     * Calculates the Mean Squared Error (MSE) between the predicted outputs and the actual outputs.
     * @param {number[][]} X - Array of input feature arrays.
     * @param {number[]} Y - Array of actual output values corresponding to the inputs.
     * @returns {number} The mean squared error between the predicted and actual outputs.
     * @throws {Error} Throws an error if the input and output arrays do not have the same length.
     */
    scoreMSE(X, Y) {
        if (X.length !== Y.length) {
            throw new Error("Input and output arrays must have the same length.");
        }
        let sumSquaredError = 0;
        for (let i = 0; i < X.length; i++) {
            const prediction = this.predict(X[i]);
            const error = Y[i] - prediction;
            sumSquaredError += error * error;
        }
        return sumSquaredError / X.length;
    }

    /**
     * Calculates the accuracy of the neuron on a dataset.
     * @param {number[][]} X - Array of input feature arrays.
     * @param {number[]} Y - Array of target outputs (0 or 1).
     * @returns {number} Accuracy as a fraction between 0 and 1.
     * @throws {Error} Throws if input arrays length mismatch.
     */
    score(X, Y) {
        if (X.length !== Y.length) {
            throw new Error("Input and output arrays must have the same length.");
        }
        let correct = 0;
        for (let i = 0; i < X.length; i++) {
            const prediction = this.predict(X[i]);
            const predictedClass = prediction >= 0.5 ? 1 : 0;
            if (predictedClass === Y[i]) {
                correct++;
            }
        }
        return correct / X.length; // accuracy between 0 and 1
    }
}

/**
 * Class representing a neural network with multiple layers and softmax output.
 */
// ... same functions above ...

class SoftmaxNN {
    constructor(inputs, hidden, outputs, learningRate = 0.01, activationType = 'sigmoid') {
      
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
        if (!outputs) {
            throw new Error('Please input a number for "output".');
        } else {
            this.output = outputs;
        }

        this.learningRate = learningRate;
        let prevLayerSize = this.inputs;
        this.layers = [];
        for (let size of this.hidden) {
           
            const layer = Array.from({ length: size }, () => new Neuron(prevLayerSize, activationType, learningRate));
            this.layers.push(layer);
            prevLayerSize = size;
        }

       
        this.outputLayer = Array.from({ length: this.output }, () => new Neuron(prevLayerSize, 'linear', learningRate));
    }

    softmax(logits) {

        const max = Math.max(...logits);
        const exps = logits.map(x => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        const probs = exps.map(e => e / sum);
       
        return probs;
    }

    predict(input) {
     
        let currentInput = input;
        let layerOutputs = [];
        for (const [i, layer] of this.layers.entries()) {
   
            currentInput = layer.map(neuron => neuron.predict(currentInput));
            layerOutputs.push(currentInput);
        }
        const outputValues = this.outputLayer.map(neuron => neuron.predict(currentInput));
      

        const softmaxOut = this.softmax(outputValues);
      
        return { output: softmaxOut, layers: layerOutputs };
    }

    train(x, y, epochs = 50) {
      
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalLoss = 0;
         
               
                for (let i = 0; i < x.length; i++) {
                    const { output: finalOutput, layers: layerOutputs } = this.predict(x[i]);
                    totalLoss += crossEntropyLoss(finalOutput, y[i]);
               console.log(finalOutput);
                    
                    
                // Output layer
                for (let j = 0; j < this.outputLayer.length; j++) {
                    const neuron = this.outputLayer[j];
                    const prevActivation = layerOutputs.length > 0 ? layerOutputs[layerOutputs.length - 1] : x[i];
                    neuron.backprop({ target: y[i][j], prevLayerActivations: prevActivation, prob: finalOutput[j] });
                }
    
                // Hidden layers
                for (let j = this.layers.length - 1; j >= 0; j--) {
                    const layer = this.layers[j];
                    const nextLayer = (j === this.layers.length - 1) ? this.outputLayer : this.layers[j + 1];
                    const prevActivations = j === 0 ? x[i] : this.layers[j - 1].map(neuron => neuron.lastOutput);
    
                    for (let k = 0; k < layer.length; k++) {
                        const neuron = layer[k];
                        let downstreamDeltaSum = 0;
                        for (const nextNeuron of nextLayer) {
                            downstreamDeltaSum += nextNeuron.delta * nextNeuron.weights[k];
                        }
                        neuron.backprop({ downstreamDeltaSum, prevLayerActivations: prevActivations });
                    }
                }
            }
    
         
    }
}

    score(x, y) {
   
        if (x.length !== y.length) {
            throw new Error("Input and output arrays must have the same length.");
        }
        let correct = 0;
        for (let i = 0; i < x.length; i++) {
            const prediction = this.predict(x[i]).output;
            const predictedClass = prediction.indexOf(Math.max(...prediction));
            const actualClass = y[i].indexOf(Math.max(...y[i]));

            if (predictedClass === actualClass) {
                correct++;
            }
        }
        const accuracy = correct / x.length;
   
        return accuracy;
    }
}

module.exports = { Neuron, SoftmaxNN };
