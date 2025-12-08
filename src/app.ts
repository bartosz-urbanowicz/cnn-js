import {Network} from "./network.js";
import {Dense} from "./layers/dense.js";
import {Input} from "./layers/input.js";

async function main(): Promise<void> {
    const network = new Network([
        new Input(4),
        new Dense(4, "sigmoid"),
        new Dense(2, "sigmoid"),
        new Dense(2, "sigmoid")
    ])


    // network.initialize()

    await network.importKerasWeights("/home/bartek/Desktop/magisterka/dl/project/cnn-ts/model.weights.h5", 4)

    // const result = network.predict([0.1, 0.2, 0.3, 0.4])
    // const gradient = network.gradient([0.1, 0.2, 0.3, 0.4], [0, 1])

    // console.log(result);
    // console.log(gradient[0]);

    network.sgd([{ data: [0.1, 0.2, 0.3, 0.4], target: [0, 1]}], 1, 0.001)

    network.printWeights()

}

main()

