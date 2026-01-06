import {multiply, add} from "mathjs";
import {sigmoid} from "../activation-functions.js";
import {Dataset, Group} from "h5wasm";
import {map} from "mathjs";
import {TrainableLayer} from './trainable-layer.ts';
import {DenseLayerParameters} from '../types/DenseLayerParameters.ts';

export class Dense extends TrainableLayer {
	public inputShape: number = 0;
	public outputShape: number = 0;
	public parameters: DenseLayerParameters = {weights: [], biases: []};

	public constructor(outputShape: number, activation: string) {
		super(activation);

		this.outputShape = outputShape;
	}

	public initialize(previousShape: number): void {
		this.inputShape = previousShape
		let limit = null;

		if (this.initializer === "xavier") {
			limit = Math.sqrt(6 / (previousShape + this.outputShape));
		} else if (this.initializer === "he") {
			limit = Math.sqrt(6 / previousShape);
		} else {
			throw new Error("select valid initializer");
		}

		this.parameters.weights = Array.from(
			{length: this.outputShape},
			() => Array.from({length: previousShape}, () => Math.random() * (2 * limit) - limit)
		);

		this.parameters.biases = Array.from({length: this.outputShape}, () => 0)
	}

	public importKerasWeights(data: Group, previousShape: number): void {
		const weightDataset: Dataset = data.get("vars/0") as Dataset
		const weights: number[] = Array.from(weightDataset.value as Int32Array)
		const biases: Dataset = data.get("vars/1") as Dataset

		this.inputShape = previousShape;

		const newWeights: number[][] = []

		for (let i = 0; i < this.outputShape; i++) {
			newWeights.push([])
			for (let j = 0; j < previousShape; j++) {
				newWeights[i].push(weights[(j * this.outputShape) + i]);
			}
		}

		this.parameters.weights = newWeights;
		this.parameters.biases = Array.from(biases.value as Int32Array)
	}


	public forward(input: number[]): number[] {
		const preActivations: number[] = add(multiply(this.parameters.weights, input), this.parameters.biases);
		const activations = this.activationFunction(preActivations);

		// stored for backprop
		this.lastPreActivations = preActivations;
		this.lastActivations = activations;

		return activations;
	}

	public activationFunctionDerivative(idx: number): number {
		if (this.activationFunctionName === "sigmoid") {
			const activation = this.lastActivations[idx]
			return activation * (1 - activation)
		}
		if (this.activationFunctionName === "relu") {
			const preActivation = this.lastPreActivations[idx]
			return preActivation > 0 ? 1 : 0;
		}
		return 0
	}

	public outputLayerWeightsGradient(losses: number[]): number[][] {
		const gradient: number[][] = []
		const deltas = []
		for (let j = 0; j < this.outputShape; j++) {
			gradient.push([])
			let delta = null;
			if (this.activationFunctionName === "sigmoid") {
				delta = losses[j] * this.activationFunctionDerivative(j);
			} else {
				delta = losses[j];
			}
			deltas.push(delta)
			for (let i = 0; i < this.inputShape; i++) {
				const changeToWeight = delta * this.previousLayer!.lastActivations[i]
				gradient[j].push(changeToWeight)
			}
		}

		this.deltas = deltas;
		return gradient
	}

	public outputLayerBiasesGradient(losses: number[]): number[] {
		const gradient: number[] = []
		for (let j = 0; j < this.outputShape; j++) {
			let changeToBias = null;
			if (this.activationFunctionName === "sigmoid") {
				changeToBias = losses[j] * this.activationFunctionDerivative(j)
			} else {
				changeToBias = losses[j];
			}

			gradient.push(changeToBias)
		}

		return gradient
	}

	public hiddenLayerWeightsGradient(): number[][] {
		const gradient: number[][] = []
		const deltas = []
		for (let j = 0; j < this.outputShape; j++) {
			gradient.push([])
			let sum = 0
			for (let k = 0; k < this.nextLayer!.parameters.weights.length; k++) {
				sum += (this.nextLayer!.parameters.weights[k][j] as number) * this.nextLayer!.deltas[k]
			}
			const delta = sum * this.activationFunctionDerivative(j)
			deltas.push(delta)
			for (let i = 0; i < this.inputShape; i++) {
				const changeToWeight = delta * this.previousLayer!.lastActivations[i]
				gradient[j].push(changeToWeight)
			}
		}

		this.deltas = deltas;
		return gradient
	}

	public hiddenLayerBiasesGradient(): number[] {
		const gradient: number[] = []
		for (let j = 0; j < this.outputShape; j++) {
			let sum = 0;
			for (let k = 0; k < this.nextLayer!.parameters.weights.length; k++) {
				sum += (this.nextLayer!.parameters.weights[k][j] as number) * this.nextLayer!.deltas[k]
			}
			const changeToBias = sum * this.activationFunctionDerivative(j)
			gradient.push(changeToBias)
		}

		return gradient
	}
}
