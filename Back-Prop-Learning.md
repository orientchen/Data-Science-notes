# Back-Prop-Learning Algorithm

**Function:** `BACK-PROP-LEARNING(examples, network)` → a neural network

**Inputs:**
- `examples`: a set of examples, each with input vector **x** and output vector **y**
- `network`: a multilayer network with \(L\) layers, weights \(w_{i,j}\), and activation function \(g\)

**Local variables:** \(\Delta\), a vector of errors indexed by network node

```text
repeat
    for each weight wᵢ,ⱼ in network do
        wᵢ,ⱼ ← a small random number

    for each example (x, y) in examples do
        /* Propagate the inputs forward to compute the outputs */
        for each node i in the input layer do
            aᵢ ← xᵢ

        for ℓ = 2 to L do
            for each node j in layer ℓ do
                inⱼ ← Σᵢ wᵢ,ⱼ aᵢ
                aⱼ ← g(inⱼ)

        /* Propagate deltas backward from output layer to input layer */
        for each node j in the output layer do
            Δ[j] ← g′(inⱼ) × (yⱼ − aⱼ)

        for ℓ = L − 1 to 1 do
            for each node i in layer ℓ do
                Δ[i] ← g′(inᵢ) Σⱼ wᵢ,ⱼ Δ[j]

        /* Update every weight in network using deltas */
        for each weight wᵢ,ⱼ in network do
            wᵢ,ⱼ ← wᵢ,ⱼ + α × aᵢ × Δ[j]

until some stopping criterion is satisfied
return network
```

## Equations used in the algorithm

**Forward propagation**

$$
\operatorname{in}_j = \sum_i w_{i,j} a_i
$$

$$
a_j = g(\operatorname{in}_j)
$$

**Output-layer error**

$$
\Delta[j] = g'(\operatorname{in}_j)(y_j-a_j)
$$

**Backward propagation of errors**

$$
\Delta[i] = g'(\operatorname{in}_i)\sum_j w_{i,j}\Delta[j]
$$

**Weight update**

$$
w_{i,j} \leftarrow w_{i,j}+\alpha a_i\Delta[j]
$$

> **Note:** The pseudocode above reproduces the supplied figure, including the placement of random weight initialization inside `repeat`. In a typical implementation, weights are initialized **once before** the training loop, rather than reinitialized on every iteration.
