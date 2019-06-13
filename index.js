//npm packages
const tf = require('@tensorflow/tfjs');
const rline = require('readline');
const json = require('./index.json');

//make an ML model
const model = tf.sequential();

//add layer
const xs = tf.tensor2d(Object.values(json).map(int => [
  int.MIN
]));
xs.print();
const ys = tf.tensor2d(Object.values(json).map(int => [
  int.MAX
]));
model.add(tf.layers.dense({
  inputShape: [1],
  units: 5,
  activation: 'relu'
}))
model.add(tf.layers.dense({
  inputShape: [5],
  units: 1,
  activation: 'sigmoid'
}))

model.add(tf.layers.dense({
  units: 1,
  activation: 'relu'
}))

model.compile({ loss: 'meanSquaredError', metrics: ['accuracy'], optimizer: tf.train.adam(0.06) });
// random data


model.fit(xs, ys, { epochs: 20 }).then(history => {
  var epochs = history.epoch;
  for (epoch in epochs) {
    console.log(`${epoch}`);
  }
  console.log(history);
  const output = model.predict(tf.tensor2d([68], [1, 1]));
  output.print()

})

