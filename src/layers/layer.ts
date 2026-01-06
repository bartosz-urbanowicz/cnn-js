import {Group} from "h5wasm";
import {Optimizer} from '../optimizers/optimizer.ts';
import {TTensorShape} from '../types/TensorShape.ts';
import {DenseLayerParameters} from '../types/DenseLayerParameters.ts';
import {Conv2dLayerParameters} from '../types/Conv2dLayerParameters.ts';

export abstract class Layer {
	public abstract inputShape: TTensorShape;
	public abstract outputShape: TTensorShape;
	protected previousLayer: null | Layer = null;
	protected nextLayer: null | Layer = null;
	public lastActivations: number[] = [];
	public lastPreActivations: number[] = [];
	public deltas: number[] = [];

	public abstract initialize(previousShape: TTensorShape): void

	public abstract forward(input: number[] | number[][][]): number[] | number[][][]

	public setPreviousLayer(layer: Layer | null): void {
		this.previousLayer = layer;
	}

	public setNextLayer(layer: Layer | null): void {
		this.nextLayer = layer;
	}
}
