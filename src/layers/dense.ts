import {Layer} from "./layer.js";
import {multiply, add} from "mathjs";
import {sigmoid} from "../activations.ts";
import {Dataset, Group} from "h5wasm";

export class Dense extends Layer {
    protected weights:number[][] = [];
    protected biases: number[] = [];
    shape: number = 0;
    private activation: (z: number[]) => number[]
    private initializer: string = "";

    constructor(shape: number, activation: string) {
        super();

        this.shape = shape;

        if (activation === "sigmoid") {
            this.activation = sigmoid;
            this.initializer = "xavier"
        } else {
            throw new Error("Provide correct activation function!")
        }

    }

    importKerasWeights(data: Group, previousShape: number): void {
        const weightDataset: Dataset = data.get("vars/0") as Dataset
        const weights: number[] = Array.from(weightDataset.value as Int32Array)
        const biases: Dataset = data.get("vars/1") as Dataset

        const newWeights: number[][] = []

        // TODO refactor transposition here
        for (let i = 0; i < this.shape; i++) {
            newWeights.push([])
            for (let j = 0; j < previousShape; j++) {
                newWeights[i].push(weights[(j * this.shape) + i]);
            }
        }

        this.weights = newWeights;
        this.biases = Array.from(biases.value as Int32Array)
    }

    initialize(previousShape: number): void{

        if (this.initializer === "xavier") {
            const x = Math.sqrt(6 / (previousShape + this.shape))

            this.weights = Array.from(
                { length: this.shape },
                () => Array.from({ length: previousShape }, () => Math.random() * (2 * x) - x)
            );

            this.biases = Array.from({ length: this.shape }, () => 0)
        }
    }

    forward(input: number[]): number[] {
        return this.activation!(add(multiply(this.weights, input), this.biases));
    }
}
