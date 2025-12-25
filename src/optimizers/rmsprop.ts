import {Network} from '../network.ts';
import {LayerSgdMomentumState} from '../types/LayerSgdMomentumState.ts';
import {Optimizer} from './optimizer.ts';
import {Layer} from '../layers/layer.ts';
import {LayerRmsPropState} from '../types/LayerRmsPropState.ts';
import { TLayerOptimizerState } from "../types/LayerOptimizerState.ts";
import {LayerAdamState} from '../types/LayerAdamState.ts';

export class RmsProp extends Optimizer {
  public state: { layers: LayerRmsPropState[] } = { layers: [] };
  private decayRate: number;
  private stabilizer: number;

  public constructor(
    learningRate: number,
    decayRate: number = 0.9,
    stabilizer: number = 0.00000001
  ) {
    super(learningRate);
    this.decayRate = decayRate;
    this.stabilizer = stabilizer;
  }

  public initializeStates(network: Network): void {
    network.layers.forEach((layer) => {
      const layerState: LayerRmsPropState = {avgSquareGradient: {weights: [], biases: []}};
      layerState.avgSquareGradient.weights = Array.from(
        { length: layer.outputShape },
        () => Array.from({ length: layer.inputShape }, () => 0)
      );

      layerState.avgSquareGradient.biases = Array.from({ length: layer.outputShape }, () => 0)

      this.state.layers.push(layerState);
    })
  };

  public applyGradient(layer: Layer, weightsGradient: number[][], biasesGradient: number[], layerIndex: number): void {
    for (let j = 0; j < layer.outputShape; j++) {
      for (let i = 0; i < layer.inputShape; i++) {
        const previousAvgSquareGradient = this.state.layers[layerIndex].avgSquareGradient.weights[j][i]
        const newAvgSquareGradient =
          this.decayRate * previousAvgSquareGradient + ((1 - this.decayRate) * Math.pow(weightsGradient[j][i], 2))
        const changeToWeight =
          (this.learningRate * weightsGradient[j][i]) / (Math.sqrt(newAvgSquareGradient) + this.stabilizer)
        layer.weights[j][i] -= changeToWeight;
        this.state.layers[layerIndex].avgSquareGradient.weights[j][i] = newAvgSquareGradient;
      }
    }

    for (let i = 0; i < layer.biases.length; i++) {
      const previousAvgSquareGradient = this.state.layers[layerIndex].avgSquareGradient.biases[i]
      const newAvgSquareGradient =
        this.decayRate * previousAvgSquareGradient + ((1 - this.decayRate) * Math.pow(biasesGradient[i], 2))
      const changeToBias =
        (this.learningRate * biasesGradient[i]) / (Math.sqrt(newAvgSquareGradient) + this.stabilizer)
      layer.biases[i] -= changeToBias;
      this.state.layers[layerIndex].avgSquareGradient.biases[i] = newAvgSquareGradient;
    }
  }


}
