import {DenseLayerParameters} from './DenseLayerParameters.ts';

export interface LayerAdamState {
  firstMoment: DenseLayerParameters;
  secondMoment: DenseLayerParameters;
}
