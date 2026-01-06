import {Network} from "../network.ts";
import {Dense} from "../layers/dense.ts";
import {Input} from "../layers/input.ts";
import {RmsProp} from '../optimizers/rmsprop.ts';
import {Sgd} from '../optimizers/sgd.ts';
import {Convolutional2d} from '../layers/convolutional-2d.ts';
import {relu} from '../activation-functions.ts';
import {Pooling2d} from '../layers/pooling-2d.ts';

async function main(): Promise<void> {
	const layer = new Convolutional2d("relu", 1, [2, 2], 1, 1)
	console.log(layer.slide(
		[[0, 1],
		[0, 1]],
		layer.pad([[1, 2, 3, 4],
		[1, 2, 3, 4],
		[1, 2, 3, 4],
		[1, 2, 3, 4]])
		))

	// const poolingLayer = new Pooling2d([2, 2], 2, "avg")
	// console.log(poolingLayer.slide([
	// 	[2, 2, 7, 3],
	// 	[9, 4, 6, 1],
	// 	[8, 5, 2, 4],
	// 	[3, 1, 2, 6]
	// ]))

}

main();

