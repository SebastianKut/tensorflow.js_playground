const loadingStatus = document.getElementById('status');
if (loadingStatus) {
  loadingStatus.innerText =
    'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}

// Generate non linear inputs and outputs
const INPUTS = [];
for (let i = 0; i <= 20; i++) {
  INPUTS.push(i);
}

const OUTPUTS = [];
for (let i = 0; i < INPUTS.length; i++) {
  OUTPUTS.push(INPUTS[i] * INPUTS[i]);
}

//Turn data arrays into tensors - input is 1d and output 1d
const INPUT_TENSOR = tf.tensor1d(INPUTS);
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

// this is input layer
model.add(
  tf.layers.dense({
    //this means each neuron in the layer is densly connceted to its inputs
    inputShape: [1], //shape represents 1 value as input to the mode
    units: 25, //up the number of neurons for better results - this is known as input layer of neurons
    activation: 'relu', // addidng activation function that is needed when the data has exponential character and non linear - it only produces the output when it crosses some output
  })
); //theres no activation function specified means output will be just passed without activation (activation function describes some threshold that we want to look at)

// We add extra layer know as hidden layer with extra 5 neurons - addding extra layers we need to decrease LEARNIN RATE
model.add(
  tf.layers.dense({
    units: 5,
    activation: 'relu',
  })
);

// Another dense layer of 1 output neuron - this is output layer - we want output layer to have no activation function as we always want it to produce some output
model.add(tf.layers.dense({ units: 1 }));

model.summary();

const LEARNING_RATE = 0.0001; //reduce learning rate by adding extra 0 as we now have more neurons

const OPTIMIZER = tf.train.sgd(LEARNING_RATE); //stochastic gradient descent is the mathematical algorithm used to update the weights at specified rate

train();

async function train() {
  // compile the model with defined learning rate and specify a loss function to use
  model.compile({
    optimizer: OPTIMIZER,
    loss: 'meanSquaredError', //common way to use loss is meanSquaredError
  });

  // Finnally do the training - passing input and output tensors as well as options
  let results = await model.fit(
    FEATURE_RESULTS.NORMALIZED_VALUES,
    OUTPUT_TENSOR,
    {
      callbacks: { onEpochEnd: logProgress }, //we can specify callback functions for different events here - I want to log out values after each epoch so we know at what stage model stops making learning progress
      shuffle: true, //shuffle data to avoid model figuring out some order, indexes of input and output will still corelate with eachother
      batchSize: 2, //not much validation data to sample so batch size is 2
      epochs: 200, // go over data 200 times
    }
  );

  OUTPUT_TENSOR.dispose();
  FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

  // Average loss shows the difference between the actuall house price and the predicetd house price
  console.log(
    'Average error loss',
    Math.sqrt(results.history.loss[results.history.loss.length - 1])
  );

  //Once trained evaluate model
  evaluate();
}

function logProgress(epoch, logs) {
  console.log('Data for epoch ' + epoch, Math.sqrt(logs.loss));
  // at 160 itteration it stops making learning progres
  if (epoch == 160) {
    //this doesnt really do much
    OPTIMIZER.setLearningRate(LEARNING_RATE / 2);
  }
}

function evaluate() {
  // predict answer for a single piece of data
  tf.tidy(function () {
    // normalize new input by passing new input that you want to predict, as well as min and max values from our training data - this is done so our input is normalized to the same values as training data
    let newInput = normalize(
      tf.tensor1d([7]),
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

//Training the model is always fine tuning the architecture changing layers number, neurons in each layers and adjusting learning rate as we increase neurons added to the model
//sometimes its better to have slightly less accurate model but much more performant
