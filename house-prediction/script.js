const loadingStatus = document.getElementById('status');
if (loadingStatus) {
  loadingStatus.innerText =
    'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}

import { TRAINING_DATA } from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js';

console.log('Training data', TRAINING_DATA.inputs);
console.log('Training data', TRAINING_DATA.outputs);

// 2D array containing house size and no of bedrooms
const INPUTS = TRAINING_DATA.inputs;

// contains 1D array of house prices
const OUTPUTS = TRAINING_DATA.outputs;

//Shuffles the 2 arrays in the same way so input indexes still match output indexes - this is done in case data was arranged in some order, to make sure that model really learns
tf.util.shuffleCombo(INPUTS, OUTPUTS);

//Turn data arrays into tensors - input is 2d and output 1d
const INPUT_TENSOR = tf.tensor2d(INPUTS);
console.log('INPUT TENSOR', INPUT_TENSOR);

const OUTPUT_TENSOR = tf.tensor1d(OUTPUTS);

// We have to normalize the data so its repeseted in the range between 0 - 1 instead of big numbers. for example 432sq ft and 3 bedrooms would be represented as something like [0.6566, 1] - this range depends on max and min values in our data set (like a percentage representation)
// min and max are optional so we can specify our own max - min range instead of looking at the whole data set
function normalize(tensor, min, max) {
  // will wrap calculations with tf.tidy that will do garbage collection for us
  const result = tf.tidy(function () {
    // Find min value contained in the tensor
    const MIN_VALUES = min || tf.min(tensor, 0); //0 represents the axis in this case in 2D tensor it will compare each column for each tensor (so same index in each array, then move to next)

    const MAX_VALUES = max || tf.max(tensor, 0);

    // Now we substract MIN_VALUE from every value in the tensor and keep it in the new tensor, again that is done by index
    const TENSOR_SUBSTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

    // Calculate rannge size of possible value by substractin min from max
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

    //To get normalized values we divide adjusted values by the range size
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBSTRACT_MIN_VALUE, RANGE_SIZE);

    // we want object that contains normalized values tensor as well as min and max values (these are not normalized)
    return {
      NORMALIZED_VALUES,
      MIN_VALUES,
      MAX_VALUES,
    };
  });

  return result;
}

// Normalize our input values
const FEATURE_RESULTS = normalize(INPUT_TENSOR);

console.log('Normalized values:');
FEATURE_RESULTS.NORMALIZED_VALUES.print();

console.log('Min values:');
FEATURE_RESULTS.MIN_VALUES.print();

console.log('Max values:');
FEATURE_RESULTS.MAX_VALUES.print();

// We dnt need INPUT_TENSOR as we have normalized version of it so we have to dispose it
INPUT_TENSOR.dispose();

// Create and define models architecture
const model = tf.sequential(); //this means that output of the layer below becomes an input for the layer above -its sequential

// we will ony have one neuron with one layer and input will be shape of 2 - size and price

model.add(
  tf.layers.dense({
    //this means each neuron in the layer is densly connceted to its inputs
    inputShape: [2], //shape represents 2 inputs to the model - size and no of bedrooms
    units: 1, //this is number of neurons - it has 1 weight allocted to each neuron which means its densly connected to its inputs
  })
); //theres no activation function specified means output will be just passed without activation (activation function describes some threshold that we want to look at)

model.summary();

train();

async function train() {
  const LEARNING_RATE = 0.01; //this has to be chosen so its suitable for the data we are using, if its too high values like nAn will appear so we would have to reduce it

  // compile the model with defined learning rate and specify a loss function to use
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE), //stochastic gradient descent is the mathematical algorithm used to update the weights at specified rate
    loss: 'meanSquaredError', //common way to use loss is meanSquaredError
  });

  // Finnally do the training - passing input and output tensors as well as options
  let results = await model.fit(
    FEATURE_RESULTS.NORMALIZED_VALUES,
    OUTPUT_TENSOR,
    {
      validationSplit: 0.15, //take 15% of input data aside for validation testing
      shuffle: true, //shuffle data to avoid model figuring out some order, indexes of input and output will still corelate with eachother
      batchSize: 64, //usually some value in the power of 2, as we have a lot of training data its set to 64 - this represents the number of examples it will go through before calculating average loss and updating wegights and bias
      epochs: 10, // go over data 10 times
    }
  );

  OUTPUT_TENSOR.dispose();
  FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

  // Average loss shows the difference between the actuall house price and the predicetd house price
  console.log(
    'Average error loss',
    Math.sqrt(results.history.loss[results.history.loss.length - 1])
  );
  console.log(
    'Average validation error loss',
    Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1])
  );

  //Once trained evaluate model
  evaluate();
}

function evaluate() {
  // predict answer for a single piece of data
  tf.tidy(function () {
    // normalize new input by passing new input that you want to predict, as well as min and max values from our training data - this is done so our input is normalized to the same values as training data
    let newInput = normalize(
      tf.tensor2d([[750, 1]]),
      FEATURE_RESULTS.MIN_VALUES,
      FEATURE_RESULTS.MAX_VALUES
    );

    let output = model.predict(newInput.NORMALIZED_VALUES);

    output.print();
  });

  // Grabage clean
  FEATURE_RESULTS.MIN_VALUES.dispose();
  FEATURE_RESULTS.MAX_VALUES.dispose();
  model.dispose();

  console.log(tf.memory().numTensors);
}

// We can call model.predict as many times as we like once the model is trained, you only have to train the model once

// We can save the model for future use
// download
// await model.save('downloads://my-model')

// save to local storage for offline access
// await model.save('localstorage://demo/newModelName')
