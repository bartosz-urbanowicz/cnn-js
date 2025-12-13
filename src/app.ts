import {Network} from "./network.js";
import {Dense} from "./layers/dense.js";
import {Input} from "./layers/input.js";
import mnist from "mnist";
import {number} from 'mathjs';

async function main(): Promise<void> {
    const model = new Network([
        new Input(784),
        new Dense(784, "sigmoid"),
        new Dense(64, "sigmoid"),
        new Dense(10, "sigmoid")
    ])

    const set = mnist.set(4000, 1000);

    const trainingSet = set.training;
    const testSet = set.test;

    const inputsTrain: number[][] = trainingSet.map((item: {input: number[], output: number[]}) => item.input);
    const outputsTrain: number[][] = trainingSet.map((item: {input: number[], output: number[]}) => item.output);

    model.initialize()

    model.fit(
        inputsTrain,
        outputsTrain,
        {
          batchSize: 32,
          learningRate: 0.001,
          epochs: 10,
          validationSplit: 0.8
        }
      )

}

main()

