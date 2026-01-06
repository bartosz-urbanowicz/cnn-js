import {DenseLayerParameters} from './DenseLayerParameters.ts';

export interface LayerRmsPropState {
  avgSquareGradient: DenseLayerParameters;
}
