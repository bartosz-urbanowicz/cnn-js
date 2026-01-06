import {Layer} from "./layer.js";

export class Flatten extends Layer {

	public inputShape: [number, number, number] = [0, 0, 0];
	public outputShape: number = 0;

	public constructor() {
		super();
	}

	public initialize(previousShape: [number, number, number]): void {
		this.inputShape = previousShape;
		this.outputShape = previousShape[0] * previousShape[1] * previousShape[2];
	}


	// tensorflow is channel-last!
	public forward(input: number[][][]): number[] {
		const result: number[] = [];
		for (let channel = 0; channel < input.length; channel++) {
			for (let i = 0; i < input[channel].length; i++) {
				result.push(...input[channel][i])
			}
		}

		console.log("flatten forward")

		return result;
	}
}
