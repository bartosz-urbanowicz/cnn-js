import {map} from "mathjs";

export function sigmoid(z: number[]): number[] {
    return map(z, (x) => 1/(1 + Math.exp(-x)))
}