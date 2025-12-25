import {Layer} from "./layer.js";
import {Group} from "h5wasm";

export class Input extends Layer {

    public constructor(inputShape: number) {
        super();

        this.inputShape = inputShape;
    }

    public initialize(previousShape: number):void {
      this.inputShape = previousShape;
      this.outputShape = previousShape;
    }

    public importKerasWeights(data: Group, previousShape: number): void {
      throw new Error("Method not implemented.");
    }

    public forward(input: number[]): number[] {
        this.lastPreActivations = input;
        this.lastActivations = input;
        return input;
    }
}
