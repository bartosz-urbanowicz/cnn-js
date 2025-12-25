import {LayerTensor} from './LayerTensor.ts';

export interface LayerRmsPropState {
  avgSquareGradient: LayerTensor;
}
