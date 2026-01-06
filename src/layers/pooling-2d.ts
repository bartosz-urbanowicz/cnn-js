import {Layer} from "./layer.js";
import {Group} from "h5wasm";
import {TTensorShape} from '../types/TensorShape.ts';

export class Pooling2d extends Layer {

	public inputShape: [number, number, number] = [0, 0, 0];
	public outputShape: [number, number, number] = [0, 0, 0];
	private poolSize: [number, number]; // height, width
	private stride: number;
	private method: string;

	public constructor(poolSize: [number, number], stride: number, method: string) {
		super();

		this.poolSize = poolSize;
		this.stride = stride;
		this.method = method;
	}

	public initialize(previousShape: [number, number, number]):void {
		this.inputShape = previousShape;

		const heightOut = Math.floor((previousShape[1] - this.poolSize[0]) / this.stride) + 1;
		const widthOut = Math.floor((previousShape[2] - this.poolSize[1]) / this.stride) + 1;

		this.outputShape = [previousShape[0], heightOut, widthOut];
	}

	private maximum(image: number[][], vertOffset: number, horOffset: number): number {
		let max = -Infinity;
		for (let i = 0; i < this.poolSize[0]; i++) {
			for (let j = 0; j < this.poolSize[1]; j++) {
				const currValue = image[i + vertOffset][j + horOffset];
				if (currValue > max) {
					max = currValue;
				}
			}
		}
		return max
	}

	private average(image: number[][], vertOffset: number, horOffset: number): number {
		let sum = 0;
		for (let i = 0; i < this.poolSize[0]; i++) {
			for (let j = 0; j < this.poolSize[1]; j++) {
				const currValue = image[i + vertOffset][j + horOffset];
				sum += currValue;
			}
		}
		return sum / (this.poolSize[0] * this.poolSize[1])
	}

	public slide(image: number[][]): number[][] {
		const result: number[][] = [[]];
		let currVertOffset = 0;
		let currHorOffset = 0;

		while (
			currHorOffset <= (image[0].length - this.poolSize[1]) &&
			currVertOffset <= (image.length - this.poolSize[0])
			) {
			if (this.method === "max") {
				result[result.length - 1].push(this.maximum(image, currVertOffset, currHorOffset))
			} else if (this.method === "avg") {
				result[result.length - 1].push(this.average(image, currVertOffset, currHorOffset))
			}

			// console.log(currHorOffset, currVertOffset, result)

			// edge reached
			if (currHorOffset + this.stride + this.poolSize[1] > image[0].length) {
				currHorOffset = 0;
				currVertOffset += this.stride;
				if (currVertOffset + this.poolSize[0] <= image.length) {
					result.push([]);
				}
			} else {
				currHorOffset += this.stride;
			}
		}

		return result;
	}

	public forward(input: number[][][]): number[][][] {
		const result: number[][][] = [];
		for (let channel = 0; channel < input.length; channel++) {
			const image = input[channel];
			const channelResult = this.slide(image);
			result.push(channelResult)
		}

		console.log("pooling forward")

		return result;
	}
}
