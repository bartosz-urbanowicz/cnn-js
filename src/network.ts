import {Layer} from "./layers/layer.js";
import {Dense} from "./layers/dense.js"
import {Group} from "h5wasm";
import {LayerGradient} from './types/LayerGradient.ts';
import {Sample} from './types/Sample.ts';

export class Network {
    private layers: Layer[]
    private trainableLayers: Dense[]

    public constructor(
        layers: Layer[]
    ) {
        this.layers = layers
        this.trainableLayers = layers.slice(1, layers.length) as Dense[]

        let previousLayer: Layer | null = null
        this.layers.forEach((layer, index) => {
          layer.setPreviousLayer(previousLayer)
          if (index != layers.length) {
            layer.setNextLayer(layers[index + 1])
          }
          previousLayer = layer;
        })
    }

    public async importKerasWeights(file: string, inputShape: number): Promise<void> {
        const h5wasm = await import("h5wasm/node");
        await h5wasm.ready;

        let f = new h5wasm.File(file, "r");

        const layers: Group = f.get("layers")! as Group

        let previousShape = inputShape
        layers.keys().forEach((layer: string, i: number) => {
            const currentLayer: Layer = this.layers[i + 1]
            currentLayer.importKerasWeights(layers.get(layer)! as Group, previousShape)
            previousShape = currentLayer.outputShape
        })
    }

    public initialize(): void {
        let previousShape = 0
        this.layers.forEach(layer => {
            layer.initialize(previousShape);
            previousShape = layer.outputShape;
        })
    }

    public predict(input: number[]): number[] {
        let output: number[] = input;
        this.layers.forEach(layer => {
            output = layer.forward(output);
        })
        return output
    }

    public gradient(batch: Sample[]): LayerGradient[] {

      const summedLosses: number[] = batch.reduce((acc: number[], curr_sample: Sample): number[] => {
        const prediction: number[] = this.predict(curr_sample.data)
        prediction.forEach((pred, idx) => {
          acc[idx] = (acc[idx] || 0) + (pred - curr_sample.target[idx])
        })
        return acc;
      }, [])

      const avgLosses = summedLosses.map(x => x / batch.length)

      const outputLayerGradients: LayerGradient = {
        weights: this.trainableLayers[this.trainableLayers.length - 1].outputLayerWeightsGradient(avgLosses),
        biases: this.trainableLayers[this.trainableLayers.length - 1].outputLayerBiasesGradient(avgLosses)
      }

      const hiddenLayersGradients: LayerGradient[] = []

      this.trainableLayers.slice(0, -1).reverse().forEach(layer => {
        hiddenLayersGradients.push({
          weights: layer.hiddenLayerWeightsGradient(),
          biases: layer.hiddenLayerBiasesGradient()
        })
      })

      return [
        ...hiddenLayersGradients.reverse(),
        outputLayerGradients
      ]
    }

    public applyGradient(gradient: LayerGradient[], learningRate: number): void {
      this.trainableLayers.forEach((layer, i) => {
        const weightsGradient = gradient[i].weights
        const biasesGradient = gradient[i].biases

        layer.applyGradient(weightsGradient, biasesGradient, learningRate)
      })
    }

    public printWeights(): void {
      this.trainableLayers.forEach((layer, i) => {
        console.log(`Layer ${i + 1}:`)
        console.log("weights:")
        console.log(layer.weights)
        console.log("biases:")
        console.log(layer.biases)
        console.log("")
      })
    }

    public sgd(data: Sample[], batchSize: number, learningRate: number): void {

      // fisher yates shuffle
      const shuffled = data.slice();
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }

      const batch = shuffled.slice(0, batchSize)

      const gradient: LayerGradient[] = this.gradient(batch);
      this.applyGradient(gradient, learningRate);
    }
}
