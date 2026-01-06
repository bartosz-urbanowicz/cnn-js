import {Network} from "../network.ts";
import {Dense} from "../layers/dense.ts";
import {Input} from "../layers/input.ts";
import {RmsProp} from '../optimizers/rmsprop.ts';
import {Sgd} from '../optimizers/sgd.ts';

async function main(): Promise<void> {
  const model = new Network([
    new Input(4),
    new Dense(4, "relu"),
    new Dense(2, "relu"),
    new Dense(2, "softmax"),
  ],
    new RmsProp(1)
    // new Sgd(0.1)
  )

  await model.importKerasWeights("/home/bartek/Desktop/magisterka/dl/project/cnn-ts/model.weights.h5", 4)
  // model.initialize()

  model.fit(
    [[0.1, 0.2, 0.3, 0.4]],
    [[0, 1]],
    {
      batchSize: 1,
      epochs: 1,
      validationSplit: 1
    }
  )

  model.printWeights()

  const prediction = model.predict([0.1, 0.2, 0.3, 0.4])
  console.log(prediction)
}

main();

