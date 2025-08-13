const {Neuron, SoftmaxNN} = require('../src/neuron');
const fs = require('fs');
const readline = require('readline');
async function loadMNIST(filePath, maxRows = 1000) { 
    const x = [];
    const y = [];
    const fileStream = fs.createReadStream(filePath);
    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity
    });
    for await (const line of rl) {
        const values = line.split(',').map(Number);
        const label = values[0];
        y.push(values[0]);
        x.push(values.slice(1).map(v => v / 255)); // Normalize pixel values
        if (x.length >= maxRows) break;
    }
    return { x, y };

}
function oneHotEncode(labels, numClasses) {
    return labels.map(label => {
      const oneHot = new Array(numClasses).fill(0);
      oneHot[label] = 1;
      return oneHot;
    });
  }
(async () => {
    const {x, y} = await loadMNIST('MNIST_CSV/mnist_train.csv', 2000);
    const {x : testX, y : testY } = await loadMNIST('MNIST_CSV/mnist_test.csv', 100);
    const learningRates = [0.001];
    const results = [];
    
    for (let lr of learningRates) {
        const model = new SoftmaxNN(784, [128, 64], 10, lr, 'tanh')
        model.train(x, oneHotEncode(y), 400); // short training for testing
        const accuracy = model.score(testX, oneHotEncode(testY, 10));
        results.push({ learningRate: lr, accuracy });
        console.log(`LR: ${lr}, Accuracy: ${accuracy}`);
    }
    
    // You can then analyze `results` to pick the best learning rate
    console.log('Results:', results);    
})()
