import {Network} from "../network.ts";
import {Dense} from "../layers/dense.ts";
import {Input} from "../layers/input.ts";
import {RmsProp} from '../optimizers/rmsprop.ts';
import {Convolutional2d} from '../layers/convolutional-2d.ts';
import {Pooling2d} from '../layers/pooling-2d.ts';
import {Flatten} from '../layers/flatten.ts';

async function main(): Promise<void> {
	const model = new Network([
			new Input([1, 8, 8]),
			new Convolutional2d("relu", 8, [2, 2], 1, 1),
			new Pooling2d([2, 2], 2, "max"),
			new Flatten(),
			new Dense(16, "relu"),
			new Dense(10, "softmax")
		],
		new RmsProp(0.001)
	)

	model.initialize()

	const prediction = model.predict(
		[[
			[1, 2, 3, 4, 5, 6, 7, 8],
			[1, 2, 3, 4, 5, 6, 7, 8],
			[1, 2, 3, 4, 5, 6, 7, 8],
			[1, 2, 3, 4, 5, 6, 7, 8],
			[1, 2, 3, 4, 5, 6, 7, 8],
			[1, 2, 3, 4, 5, 6, 7, 8],
			[1, 2, 3, 4, 5, 6, 7, 8],
			[1, 2, 3, 4, 5, 6, 7, 8],
		]]
	)

	console.log(prediction)
}

main()

