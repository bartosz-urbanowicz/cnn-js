import {Network} from '../network.ts';
import {LayerSgdMomentumState} from '../types/LayerSgdMomentumState.ts';
import {Optimizer} from './optimizer.ts';
import {Layer} from '../layers/layer.ts';

export class Sgd extends Optimizer{

  public state: {} = {};

  public constructor(learningRate: number) {
    super(learningRate);
  }

  public initializeStates(network: Network): void {};

  public applyGradient(layer: Layer, weightsGradient: number[][], biasesGradient: number[]): void {
    for (let j = 0; j < layer.outputShape; j++) {
      for (let i = 0; i < layer.inputShape; i++) {
        const changeToWeight = this.learningRate * weightsGradient[j][i];
        layer.weights[j][i] -= changeToWeight;
      }
    }

    for (let i = 0; i < layer.biases.length; i++) {
      const changeToBias = this.learningRate * biasesGradient[i];
      layer.biases[i] -= changeToBias;
    }
  }

}
