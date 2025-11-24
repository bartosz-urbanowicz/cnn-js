import {Network} from "./network";
import {Dense} from "./layers/dense";
import {Input} from "./layers/input";

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

    const result = network.feedForward([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    console.log(result)

}


main()

