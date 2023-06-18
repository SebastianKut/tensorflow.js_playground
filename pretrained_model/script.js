const MODEL_PATH =
  'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json';

let model = null;

const loadModel = async () => {
  // there are 2 types of models - layer and grid - layer are easier to use, grid are more low level and very performant
  model = await tf.loadLayersModel(MODEL_PATH);
  const summary = model.summary();

  console.log(summary);
  // summarry tells you about nput shape for example
  // [null, 1] means input expeted is one number and null means batch of any size can be passed as input - to pass values as batch, you have to use tensor thats one D higher
  // so this model can take for example shape that has 3 batches of one value like this [3,1] but woulndt take [1,3] (which represents 1 batch of 3)

  // Batch means that instead of invoking predict function 3 times passing single input each time we can pass batch of 3 inputs
  // Create input batch that only has one element - the inut value is 870 passed as 2D tensor
  const firstBatch = tf.tensor2d([[870]]);

  // batch f 3 input values
  const secondBatch = tf.tensor2d([[500], [1100], [970]]);

  // Predict
  const resultFirstBatch = model.predict(firstBatch);
  const resultSecondBatch = model.predict(secondBatch);

  // Print to the consol of use .array() to get results back as an array to further modify
  resultFirstBatch.print();
  resultSecondBatch.print();

  // With tensr flow you have to do your own garbage cleaning
  firstBatch.dispose();
  secondBatch.dispose();
  resultFirstBatch.dispose();
  resultSecondBatch.dispose();
  model.dispose();
};

loadModel();
