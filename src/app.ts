import {Network} from "./network.js";
import {Dense} from "./layers/dense.js";
import {Input} from "./layers/input.js";

async function main() {
    const network = new Network([
        new Input(10),
        new Dense(10, "sigmoid"),
        new Dense(20, "sigmoid"),
        new Dense(30, "sigmoid"),
        new Dense(20, "sigmoid"),
        new Dense(2, "sigmoid")
    ])

    // network.initialize()

    await network.importKerasWeights("/home/bartek/Desktop/magisterka/dl/project/cnn-ts/model.weights.h5")

    const result = network.predict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    console.log(result)

}


main()

