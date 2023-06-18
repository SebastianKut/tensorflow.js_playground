const loadingStatus = document.getElementById('status');
if (loadingStatus) {
  loadingStatus.innerText =
    'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}

const MODEL_PATH =
  'https://tfhub.dev/google/tfjs-model/movenet/multipose/lightning/1';

const EXAMPLE_IMG = document.getElementById('exampleImg');

let movenet = null;

const loadAndRunModel = async () => {
  movenet = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true });
  // example tenspr is made of zeros - we passing shape to it
  let exampleInputTensor = tf.zeros([1, 256, 256, 3], 'int32');
  // passing imagge to a tf function that will turn it into a tensor
  let imageTensor = tf.browser.fromPixels(EXAMPLE_IMG);
  console.log(imageTensor.shape);

  // Cropping the image to be a square - we could use object detection model to crop an image around the person, in this example we just take the correct numbers and pretend it was done already
  let cropStartPoint = [15, 170, 0]; //this represents [Y, X, Starting color channel - 0 to make sure we have all the colors captured]
  let cropSize = [345, 345, 3]; // [height px, width px, color channels - all 3 color channels]
  let croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize);

  let resizedTensor = tf.image
    .resizeBilinear(croppedTensor, [256, 256], true) //croppedTensorImage, size in px, true is recommended, defines method of scaling corner pixels
    .toInt(); // cast all the numbers to be integers as model expects inputs to be integers not floats
  console.log(resizedTensor.shape); // the shape of resized tensor is now [256,256, 3]

  //model expects the input shape to be [1, 256,256,3] so we need to expand the shape (dimension) by adding first number which is batch - see model docs
  let tensorOutput = movenet.predict(tf.expandDims(resizedTensor));
  let arrayOutput = await tensorOutput.array();

  console.log(arrayOutput);
  //in returned 6 arrays of 56 elements (each representing one person - it can detect up to 3 ppl the first one is our person in question)
  //to get proper coordinates we should take first item in an array that represents Y coordinate and multiply by 256 (the height of our array) and second item by 256 (represents the width) then swap them around to get x, y coordinates that represents where the nose is
  // it would be Y - 24, X - 124 so (124, 24)

  //Last thing to do is take the locations from our square 256, 256 image back to the original one that is
  // we can multiply the found predicted coordinates by 345 and then add starting coordinates of 170, 15 to overlay coordiantes properly
  // so Y would be 46, X would be 337

  // Last thing to do is clean up tensors and models
};

loadAndRunModel();
