import {Layer} from "./layer.js";
import {Group} from "h5wasm";
import {TrainableLayer} from './trainable-layer.ts';
import {Conv2dLayerParameters} from '../types/Conv2dLayerParameters.ts';
import {add, sum} from 'mathjs';

export class Convolutional2d extends TrainableLayer {
	public inputShape: [number, number, number] = [0, 0, 0]; // channels, height, width
	public outputShape: [number, number, number] = [0, 0, 0]; // channels, height, width
	private filters: number;
	private kernelSize: [number, number]; // height, width
	private padding: number;
	public parameters: Conv2dLayerParameters = { weights: [], biases: [] };
	private stride: number;

	public constructor(
		activation: string,
		filters: number, // out channels
		kernelSize: [number, number],
		padding: number,
		stride: number
	) {
		super(activation);

		this.filters = filters;
		this.kernelSize = kernelSize;
		this.padding = padding;
		this.stride = stride;
	}

	public initialize(previousShape: [number, number, number]): void {
		this.inputShape = previousShape

		const heightOut = Math.floor((previousShape[1] + 2 * this.padding - this.kernelSize[0]) / this.stride) + 1;
		const widthOut = Math.floor((previousShape[2] + 2 * this.padding - this.kernelSize[1]) / this.stride) + 1;

		this.outputShape = [this.filters, heightOut, widthOut];

		let limit = null;
		if (this.initializer === "xavier") {
			limit = Math.sqrt(6 / (
				(previousShape[0] * this.kernelSize[0] * this.kernelSize[1]) +
				(this.filters * this.kernelSize[0] * this.kernelSize[1])
			));
		} else if (this.initializer === "he") {
			limit = Math.sqrt(6 / (previousShape[0] * this.kernelSize[0] * this.kernelSize[1]));
		} else {
			throw new Error("select valid initializer");
		}

		this.parameters.weights = Array.from(
			{length: this.filters},
			() => Array.from(
				{length: previousShape[0]},
				() => Array.from(
					{length: this.kernelSize[0]},
					() => Array.from(
						{length: this.kernelSize[1]},
						() => Math.random() * (2 * limit) - limit)
				)
			)
		);

		this.parameters.biases = Array.from({length: this.filters}, () => 0)
	}

	public activationFunctionDerivative(idx: number): number {
		const preActivation = this.lastPreActivations[idx]
		return preActivation > 0 ? 1 : 0;
	}

	public hiddenLayerWeightsGradient(): number[][] {
		return [];
	}

	public hiddenLayerBiasesGradient(): number[] {
		return [];
	}

	public importKerasWeights(data: Group, previousShape: number): void {
		throw new Error("Method not implemented.");
	}

	public pad(image: number[][]): number[][] {
		const result: number[][] = [];
		// top
		for (let i = 0; i < this.padding; i++) {
			result.push(Array(image[0].length + 2 * this.padding).fill(0))
		}
		// center
		for (let i = 0; i < image.length; i++) {
			result.push(Array(this.padding).fill(0).concat(image[i], Array(this.padding).fill(0)))
		}
		// bottom
		for (let i = 0; i < this.padding; i++) {
			result.push(Array(image[0].length + 2 * this.padding).fill(0))
		}

		return result;
	}

	public slide(filter: number[][], image: number[][]): number[][] {
		const result: number[][] = [[]];
		let currVertOffset = 0;
		let currHorOffset = 0;

		while (
			true
			) {

			if (
				currVertOffset + this.kernelSize[0] > image.length ||
				currHorOffset + this.kernelSize[1] > image[0].length
			) {
				break;
			}

			let sum = 0;
			filter.forEach((row, i) => {
				row.forEach((value, j) => {
					const currValue = image[i + currVertOffset][j + currHorOffset];
					sum += currValue * value;
				})
			})
			result[result.length - 1].push(sum)

			// console.log(currHorOffset, currVertOffset, result)

			// edge reached
			if (currHorOffset + this.stride + this.kernelSize[1] > image[0].length) {
				currHorOffset = 0;
				currVertOffset += this.stride;
				if (currVertOffset + this.kernelSize[0] <= image.length) {
					result.push([]);
				}
			} else {
				currHorOffset += this.stride;
			}
		}

		return result;
	}

	public forward(input: number[][][]): number[][][] {
		const preActivations: number[][][] = [];
		for (let filter = 0; filter < this.filters; filter++) {
			const kernel = this.parameters.weights[filter][0];
			const paddedImage = this.pad(input[0]);

			const outH = Math.floor(
				(paddedImage.length - kernel.length) / this.stride
			) + 1;

			const outW = Math.floor(
				(paddedImage[0].length - kernel[0].length) / this.stride
			) + 1;


			let filterResult: number[][] = Array.from(
				{ length: outH },
				() => Array(outW).fill(0)
			);

			for (let channel = 0; channel < input.length; channel++) {
				const image = this.pad(input[channel]);
				const channelResult = this.slide(this.parameters.weights[filter][channel], image);
				filterResult = add(filterResult, channelResult);
			}
			const bias = this.parameters.biases[filter];
			for (let i = 0; i < filterResult.length; i++) {
				for (let j = 0; j < filterResult[0].length; j++) {
					filterResult[i][j] += bias;
				}
			}
			preActivations.push(filterResult);
		}

		// TODO allow other functions than relu

		const activations: number[][][] = [];
		for (let i = 0; i < preActivations.length; i++) {
			activations.push([])
			for (let j = 0; j < preActivations[0].length; j++) {
				activations[activations.length - 1].push(preActivations[i][j].map((x) => Math.max(0, x)))
			}
		}

		// this.lastPreActivations = preActivations;
		// this.lastActivations = activations;

		console.log("conv forward")

		return activations;
	}
}
