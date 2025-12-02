import {Layer} from "./layer.js";
import {Group} from "h5wasm";

export class Input extends Layer {

    shape: number;

    constructor(inputShape: number) {
        super();

        this.shape = inputShape;
    }

    initialize() {

    }

    importKerasWeights(data: Group, previousShape: number) {

    }

    forward(input: number[]): number[] {
        return input;
    }
}
