import {Layer} from "./layer.js";
import {Group} from "h5wasm";

export class Input extends Layer {
    public activationFunctionDerivative(x: number): number {
        throw new Error("Method not implemented.");
    }

    public shape: number;

    public constructor(inputShape: number) {
        super();

        this.shape = inputShape;
    }

    public initialize():void {
      throw new Error("Method not implemented.");
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
