import {Layer} from "./layers/layer.js";
import {Dense} from "./layers/dense.js"
import {Group} from "h5wasm";
import {LayerGradient} from './types/LayerGradient.ts';
import {Sample} from './types/Sample.ts';
import {NetworkParams} from './types/NetworkParams.ts';
import {accuracy} from './layers/metrics.ts';

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
          if (index != layers.length - 1) {
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
        let previousShape = this.layers[0].inputShape
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

      this.trainableLayers.slice(0, -1).reverse().forEach((layer) => {
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
        // console.log(`layer ${i}`)
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

    public sgd(
      trainingData: Sample[],
      validationData: Sample[],
      batchSize: number,
      learningRate: number): void {

      // fisher yates shuffle
      const shuffled = trainingData.slice();
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }

      const numBatches: number = Math.ceil(trainingData.length / batchSize)

      for (let i = 0; i < numBatches; i++) {
        // replacing the last line
        process.stdout.write(`batch ${i}/${numBatches}\r`)
        const start = i * batchSize;
        const end = Math.min(start + batchSize, shuffled.length);
        const batch = shuffled.slice(start, end);
        const gradient: LayerGradient[] = this.gradient(batch);
        this.applyGradient(gradient, learningRate);
      }

      const acc: number = accuracy(this, validationData)
      console.log(`val_accuracy: ${acc}`)
    }

    public fit(inputs: number[][], outputs: number[][], params: NetworkParams): void{
      const data: Sample[] = inputs.map((input, index) => ({
        data: input,
        target: outputs[index]
      }));

      const shuffled = data.slice();
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }

      const splitIndex = Math.floor(data.length * params.validationSplit);
      const trainingData = data.slice(0, splitIndex);
      const validationData = data.slice(splitIndex);

      for (let i = 0; i < params.epochs; i++) {
        console.log(`Epoch ${i}/${params.epochs}`)
        this.sgd(trainingData, validationData, params.batchSize, params.learningRate)
      }
    }
}
