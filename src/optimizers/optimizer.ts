import {Network} from '../network.ts';
import {Layer} from '../layers/layer.ts';
import {TLayerOptimizerState} from '../types/LayerOptimizerState.ts';


export abstract class Optimizer {

  public constructor(learningRate: number) {
    this.learningRate = learningRate;
  }

  public abstract state: TLayerOptimizerState;

  protected learningRate: number = 0;

  public abstract initializeStates(network: Network): void;

  // reference to layer is passed to update velocities directly here
  // public abstract changeToParam(layer: Layer, gradientValue: number): number;

  public abstract applyGradient(layer: Layer, weightsGradient: number[][], biasesGradient: number[], layerIndex: number): void
}
