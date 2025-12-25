import {LayerTensor} from './LayerTensor.ts';

export interface LayerAdamState {
  firstMoment: LayerTensor;
  secondMoment: LayerTensor;
}
