import {LayerSgdMomentumState} from './LayerSgdMomentumState.ts';
import {LayerAdamState} from './LayerAdamState.ts';
import {LayerRmsPropState} from './LayerRmsPropState.ts';

export type TLayerOptimizerState = LayerAdamState | LayerSgdMomentumState | LayerRmsPropState | {};
