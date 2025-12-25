import {LayerTensor} from './LayerTensor.ts';

export interface LayerSgdMomentumState {
  velocity: LayerTensor;
}
