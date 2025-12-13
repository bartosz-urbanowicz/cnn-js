import {Layer} from "./layer.js";
import {multiply, add} from "mathjs";
import {sigmoid} from "../activation-functions.js";
import {Dataset, Group} from "h5wasm";
import {map} from "mathjs";

export class Dense extends Layer {
    public weights:number[][] = [];
    public biases: number[] = [];
    private activationFunction: (x: number) => number
    private activationFunctionName: string
    private initializer: string = "";

    public constructor(shape: number, activation: string) {
        super();

        this.outputShape = shape;

        if (activation === "sigmoid") {
            this.activationFunction = sigmoid;
            this.activationFunctionName = "sigmoid";
            this.initializer = "xavier"
        } else {
            throw new Error("Provide correct activation function!")
        }

    }

    public importKerasWeights(data: Group, previousShape: number): void {
        const weightDataset: Dataset = data.get("vars/0") as Dataset
        const weights: number[] = Array.from(weightDataset.value as Int32Array)
        const biases: Dataset = data.get("vars/1") as Dataset

        this.inputShape = previousShape;

        const newWeights: number[][] = []

        for (let i = 0; i < this.outputShape; i++) {
            newWeights.push([])
            for (let j = 0; j < previousShape; j++) {
                newWeights[i].push(weights[(j * this.outputShape) + i]);
            }
        }

        this.weights = newWeights;
        this.biases = Array.from(biases.value as Int32Array)
    }

    public initialize(previousShape: number): void{

        this.inputShape = previousShape

        if (this.initializer === "xavier") {
            const x = Math.sqrt(6 / (previousShape + this.outputShape))

            this.weights = Array.from(
                { length: this.outputShape },
                () => Array.from({ length: previousShape }, () => Math.random() * (2 * x) - x)
            );

            this.biases = Array.from({ length: this.outputShape }, () => 0)
        }
    }

    public forward(input: number[]): number[] {
        const preActivations: number[] = add(multiply(this.weights, input), this.biases);
        const activations = map(preActivations, (x) => this.activationFunction(x));

        // stored for backprop
        this.lastPreActivations = preActivations;
        this.lastActivations = activations;

        return activations;
    }

    public activationFunctionDerivative(idx: number): number {
      const activation = this.lastActivations[idx]
      if (this.activationFunctionName === "sigmoid") {
        return activation * (1 - activation)
      }
      return 0
    }

    public outputLayerWeightsGradient(losses: number[]): number[][] {
      const gradient: number[][] = []
      const deltas = []
      for (let j = 0; j < this.outputShape; j++) {
        gradient.push([])
        const delta = losses[j] * this.activationFunctionDerivative(j)
        deltas.push(delta)
        for (let i = 0; i < this.inputShape; i++) {
          const changeToWeight = delta * this.previousLayer!.lastActivations[i]
          gradient[j].push(changeToWeight)
        }
      }

      this.deltas = deltas;
      return gradient
    }

    public outputLayerBiasesGradient(losses: number[]): number[] {
      const gradient: number[] = []
      for (let j = 0; j < this.outputShape; j++) {
        const changeToBias = losses[j] * this.activationFunctionDerivative(j)
        gradient.push(changeToBias)
      }

      return gradient
    }

  public hiddenLayerWeightsGradient(): number[][] {
    const gradient: number[][] = []
    const deltas = []
    for (let j = 0; j < this.outputShape; j++) {
      gradient.push([])
      let sum = 0
      for (let k = 0; k < this.nextLayer!.weights.length; k++) {
        sum += this.nextLayer!.weights[k][j] * this.nextLayer!.deltas[k]
      }
      const delta = sum * this.activationFunctionDerivative(j)
      deltas.push(delta)
      // console.log(this.previousLayer!.lastActivations)
      for (let i = 0; i < this.inputShape; i++) {
        const changeToWeight = delta * this.previousLayer!.lastActivations[i]
        gradient[j].push(changeToWeight)
      }
    }

    this.deltas = deltas;
    return gradient
  }

  public hiddenLayerBiasesGradient(): number[] {
    const gradient: number[] = []
    for (let j = 0; j < this.outputShape; j++) {
      let sum = 0;
      for (let k = 0; k < this.nextLayer!.weights.length; k++) {
        sum += this.nextLayer!.weights[k][j] * this.nextLayer!.deltas[k]
      }
      const changeToBias = sum * this.activationFunctionDerivative(j)
      gradient.push(changeToBias)
    }

    return gradient
  }

  public applyGradient(weightsGradient: number[][], biasesGradient: number[], learningRate: number): void {
    let count = 0;
    for (const row of weightsGradient) {
      for (const val of row) {
        if (val === 0) count++;
      }
    }
    // console.log(count)
    for (let j = 0; j < this.outputShape; j++) {
      for (let i = 0; i < this.inputShape; i++) {
        this.weights[j][i] -= (weightsGradient[j][i] * learningRate)
      }
    }

    for (let i = 0; i < this.biases.length; i++) {
      this.biases[i] -= (biasesGradient[i] * learningRate)
    }
  }


}
