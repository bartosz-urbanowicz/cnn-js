
export function sigmoid(x: number): number {
    // return map(z, (x) => 1/(1 + Math.exp(-x)))
  return  1/(1 + Math.exp(-x))
}
