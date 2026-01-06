import {Network} from "../network.ts";
import {Dense} from "../layers/dense.ts";
import {Input} from "../layers/input.ts";
// @ts-ignore
import mnist from "mnist";
import {SgdMomentum} from '../optimizers/sgd-momentum.ts';
import {Sgd} from '../optimizers/sgd.ts';
import {RmsProp} from '../optimizers/rmsprop.ts';
import {Adam} from '../optimizers/adam.ts';

async function main(): Promise<void> {
	const model = new Network([
			new Input(784),
			new Dense(784, "relu"),
			new Dense(64, "relu"),
			new Dense(10, "softmax")
		],
		// new Sgd(0.001)
		// new SgdMomentum(0.1)
		// best for mnist
		new RmsProp(0.001)
		// new Adam(0.001)
	)

	const set = mnist.set(60000, 10000);

	const trainingSet = set.training;
	const testSet = set.test;

	const inputsTrain: number[][] = trainingSet.map((item: { input: number[], output: number[] }) => item.input);
	const outputsTrain: number[][] = trainingSet.map((item: { input: number[], output: number[] }) => item.output);

	model.initialize()

	model.fit(
		inputsTrain,
		outputsTrain,
		{
			batchSize: 32,
			epochs: 10,
			validationSplit: 0.8
		}
	)

}

main()

