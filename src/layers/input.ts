import {Layer} from "./layer.js";
import {Group} from "h5wasm";
import {TTensorShape} from '../types/TensorShape.ts';

export class Input extends Layer {

	public inputShape: TTensorShape;
	public outputShape: TTensorShape;

    public constructor(inputShape: TTensorShape) {
        super();

        this.inputShape = inputShape;
        this.outputShape = inputShape;
    }

    public initialize(previousShape: number):void {
      this.inputShape = previousShape;
      this.outputShape = previousShape;
    }

    public forward(input: number[]): number[] {
        this.lastPreActivations = input;
        this.lastActivations = input;
        return input;
    }
}
