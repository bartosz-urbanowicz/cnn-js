import {Layer} from "./layers/layer.js";
import {Dense} from "./layers/dense.js"
import {Group} from "h5wasm";
import {LayerTensor} from './types/LayerTensor.ts';
import {Sample} from './types/Sample.ts';
import {NetworkParams} from './types/NetworkParams.ts';
import {accuracy} from './metrics.ts';
import {Optimizer} from './optimizers/optimizer.ts';
import {TrainableLayer} from './layers/trainable-layer.ts';

export class Network {
    public layers: Layer[]
    private trainableLayers: TrainableLayer[]
    private optimizer: Optimizer;

    public constructor(
        layers: Layer[],
        optimizer: Optimizer
    ) {
        this.layers = layers
        this.trainableLayers = layers.slice(1, layers.length) as TrainableLayer[]

        this.trainableLayers.forEach((layer: TrainableLayer) => {
          layer.optimizer = optimizer;
        })

        let previousLayer: Layer | null = null
        this.layers.forEach((layer, index) => {
          layer.setPreviousLayer(previousLayer)
          if (index != layers.length - 1) {
            layer.setNextLayer(layers[index + 1])
          }
          previousLayer = layer;
        })

        this.optimizer = optimizer;
    }

    public async importKerasWeights(file: string, inputShape: number): Promise<void> {
        const h5wasm = await import("h5wasm/node");
        await h5wasm.ready;

        let f = new h5wasm.File(file, "r");

        const layers: Group = f.get("layers")! as Group


        let previousShape = inputShape
        // input layers weights are not imported, but it has to be initialized
        this.layers[0].initialize(previousShape)

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

    public gradient(batch: Sample[]): LayerTensor[] {
      const gradientSum: LayerTensor[] = this.trainableLayers.map(layer => ({
        weights: Array.from({ length: layer.outputShape },
          () => Array(layer.inputShape).fill(0)),
        biases: Array(layer.outputShape).fill(0)
      }))

      for (const sample of batch) {
        const prediction: number[] = this.predict(sample.data)
        const losses = prediction.map((pred, idx) => {
          return pred - sample.target[idx]
        })

        const outputLayerGradients: LayerTensor = {
          weights: this.trainableLayers[this.trainableLayers.length - 1].outputLayerWeightsGradient(losses),
          biases: this.trainableLayers[this.trainableLayers.length - 1].outputLayerBiasesGradient(losses)
        }

        const hiddenLayersGradients: LayerTensor[] = []

        this.trainableLayers.slice(0, -1).reverse().forEach((layer) => {
          hiddenLayersGradients.push({
            weights: layer.hiddenLayerWeightsGradient(),
            biases: layer.hiddenLayerBiasesGradient()
          })
        })

        const singleGradient: LayerTensor[] = [
          ...hiddenLayersGradients.reverse(),
          outputLayerGradients
        ]

        singleGradient.forEach((layer: LayerTensor, idx: number) => {
          for (let j = 0; j < layer.weights.length; j++) {
            for (let i = 0; i < layer.weights[j].length; i++) {
              gradientSum[idx].weights[j][i] += layer.weights[j][i]
            }
            gradientSum[idx].biases[j] += layer.biases[j]
          }
        })
      }

      // avg
      gradientSum.forEach(layer => {
        for (let j = 0; j < layer.weights.length; j++) {
          for (let i = 0; i < layer.weights[j].length; i++) {
            layer.weights[j][i] /= batch.length
          }
          layer.biases[j] /= batch.length
        }
      })

      return gradientSum;

    }

    public applyGradient(gradient: LayerTensor[]): void {
      // only for adam
      // this.trainableLayers[0].optimizer.state.timestep! += 1;
      this.trainableLayers.forEach((layer, i) => {
        const weightsGradient = gradient[i].weights
        const biasesGradient = gradient[i].biases
        layer.optimizer!.applyGradient(layer, weightsGradient, biasesGradient, i)
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
      network: Network,
      trainingData: Sample[],
      validationData: Sample[],
      batchSize: number,
    ): void {

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
        const gradient: LayerTensor[] = network.gradient(batch);
        network.applyGradient(gradient);
      }

      const acc: number = accuracy(network, validationData)
      console.log(`val_accuracy: ${acc}`)
    }

    public fit(inputs: number[][], outputs: number[][], params: NetworkParams): void{
      this.optimizer.initializeStates(this);

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
        console.log(`Epoch ${i + 1}/${params.epochs}`)
        this.sgd(this, trainingData, validationData, params.batchSize)
      }
    }
}
