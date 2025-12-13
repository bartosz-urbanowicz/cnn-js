import {Group} from "h5wasm";
import {LayerGradient} from '../types/LayerGradient.ts';

export abstract class Layer {
    public inputShape: number = 0;
    public outputShape: number = 0;
    protected previousLayer: null | Layer = null;
    protected nextLayer: null | Layer = null;
    public lastActivations: number[] = [];
    public lastPreActivations: number[] = [];
    public weights:number[][] = [];
    public biases: number[] = [];
    public deltas: number[] = [];

    public abstract importKerasWeights(data: Group, previousShape: number): void

    public abstract initialize(previousShape: number): void

    public abstract forward(input: number[]): number[]

    public abstract applyGradient(weightsGradient: number[][], biasesGradient: number[], learningRate: number): void

    public abstract activationFunctionDerivative(x: number): number

    public setPreviousLayer(layer: Layer | null): void {
      this.previousLayer = layer;
    }

    public setNextLayer(layer: Layer | null): void {
      this.nextLayer = layer;
    }
}
