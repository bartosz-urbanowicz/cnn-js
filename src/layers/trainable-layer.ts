import {Group} from "h5wasm";
import {Optimizer} from '../optimizers/optimizer.ts';
import {Layer} from './layer.ts';
import {relu, sigmoid, softmax} from '../activation-functions.ts';

export abstract class TrainableLayer extends Layer{
  //TODO refactor to use layer tensor
  public weights:number[][] = [];
  public biases: number[] = [];
  protected activationFunction: (x: number[]) => number[];
  protected activationFunctionName: string;
  protected initializer: string = "";
  // can be undefined because added in network constructor for every layer
  public optimizer: Optimizer | undefined;

  public constructor(activation: string) {
    super();

    if (activation === "sigmoid") {
      this.activationFunction = sigmoid;
      this.activationFunctionName = "sigmoid";
      this.initializer = "xavier"
    } else if (activation === "softmax") {
      this.activationFunction = softmax;
      this.activationFunctionName = "softmax";
      this.initializer = "xavier"
    } else if (activation === "relu") {
      this.activationFunction = relu;
      this.activationFunctionName = "relu";
      this.initializer = "he"
    }
    else {
      throw new Error("Provide correct activation function!")
    }

  }

  public abstract activationFunctionDerivative(x: number): number;

  public abstract outputLayerWeightsGradient(losses: number[]): number[][];

  public abstract outputLayerBiasesGradient(losses: number[]): number[];

  public abstract hiddenLayerWeightsGradient(): number[][];

  public abstract hiddenLayerBiasesGradient(): number[];
}
