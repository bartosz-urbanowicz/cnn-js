import {Group} from "h5wasm";

export abstract class Layer {
    shape: number = 0;

    abstract importKerasWeights(data: Group, previousShape: number): void

    abstract initialize(previousShape: number): void

    abstract forward(input: number[]): number[]
}