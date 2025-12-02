import {Layer} from "./layers/layer.js";
import {Dataset, Group} from "h5wasm";

export class Network {
    private layers: Layer[]

    constructor(
        layers: Layer[]
    ) {
        this.layers = layers
    }

    async importKerasWeights(file: string): Promise<void> {
        const h5wasm = await import("h5wasm/node");
        await h5wasm.ready;

        let f = new h5wasm.File(file, "r");

        const layers: Group = f.get("layers")! as Group

        let previousShape = 10
        layers.keys().forEach((layer: string, i: number) => {
            const currentLayer: Layer = this.layers[i + 1]
            currentLayer.importKerasWeights(layers.get(layer)! as Group, previousShape)
            previousShape = currentLayer.shape
        })
    }

    initialize(): void {
        let previousShape = 0
        this.layers.forEach(layer => {
            layer.initialize(previousShape);
            previousShape = layer.shape;
        })
    }

    predict(input: number[]): number[] {
        let output: number[] = input;
        this.layers.forEach(layer => {
            output = layer.forward(output);
        })
        return output
    }
}
