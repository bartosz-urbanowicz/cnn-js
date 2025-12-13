import {Network} from "./network.js";
import {Dense} from "./layers/dense.js";
import {Input} from "./layers/input.js";

async function main(): Promise<void> {
  const model = new Network([
    new Input(4),
    new Dense(4, "sigmoid"),
    new Dense(2, "sigmoid"),
    new Dense(2, "sigmoid")
  ])

  await model.importKerasWeights("/home/bartek/Desktop/magisterka/dl/project/cnn-ts/model.weights.h5", 4)
  // model.initialize()

  // model.sgd([{data: [0.1, 0.2, 0.3, 0.4], target: [0, 1]}], [{data: [0.1, 0.2, 0.3, 0.4], target: [0, 1]}],1, 0.001)

  model.fit(
    [[0.1, 0.2, 0.3, 0.4]],
    [[0, 1]],
    {
      batchSize: 1,
      learningRate: 0.001,
      epochs: 1,
      validationSplit: 1
    }
  )

  model.printWeights()
}

main();

