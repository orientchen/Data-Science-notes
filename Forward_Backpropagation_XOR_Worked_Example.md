---
title: "Forward Propagation and Backpropagation"
subtitle: "A worked XOR neural-network example"
author: "Supplementary Data Science lecture notes"
date: ""
geometry: margin=0.82in
fontsize: 10pt
header-includes:
  - \usepackage{amsmath}
  - \usepackage{booktabs}
  - \usepackage{microtype}
---

# Purpose and connection to zyBooks

The zyBooks multilayer perceptron activity illustrates the **direction** of information flow: forward propagation computes a prediction and loss; backward propagation computes how the weights should change. This note works through the arithmetic that the animation omits.

The numerical example is adapted from the uploaded **“ANN Numerical Example x1XORx2 000”**: a network trained on the first XOR input, $(x_1,x_2)=(0,0)$, with target $y=0$. It follows the same node numbering, initial weights, sigmoid activation, learning rate, and AIMA-style delta notation.

> **Scope:** This is **one training step on one example**, not a demonstration that the entire XOR function has been learned. To learn XOR, the network must also train on $(0,1)$, $(1,0)$, and $(1,1)$ over repeated passes.

# 1. Understand the network

The network has two input features, two hidden neurons, and one output neuron.

- Inputs: $x_1$ and $x_2$.
- Bias input: $x_0=1$ (the bias is represented as a weight from a constant input).
- Hidden neurons: nodes **3** and **4**, with activations $a_3$ and $a_4$.
- Output neuron: node **5**, with activation $a_5=\hat y$.
- Learning rate: $\alpha=0.1$.
- Every initial weight equals $0.1$.

**Figure 1. Neural network structure used in this example.**

![Network topology: bias input x0=1 connects to hidden nodes 3 and 4 and output node 5 through w03, w04 and w05; x1 connects to hidden nodes 3 and 4 through w13 and w14; x2 connects through w23 and w24; hidden nodes connect to output node 5 through w35 and w45. All nine directed links are labeled.](images/ANN_Accessible_Network_Diagram.svg)

*Figure 1. Two-input, two-hidden-neuron, one-output-neuron network, including the constant bias input. Arrows indicate the direction of forward propagation.*

The connections are:

| Destination | Incoming connections |
|:--|:--|
| Hidden node 3 | $w_{03}$ (bias), $w_{13}$ (from $x_1$), $w_{23}$ (from $x_2$) |
| Hidden node 4 | $w_{04}$ (bias), $w_{14}$ (from $x_1$), $w_{24}$ (from $x_2$) |
| Output node 5 | $w_{05}$ (bias), $w_{35}$ (from node 3), $w_{45}$ (from node 4) |

The subscript $w_{ij}$ means **weight on the connection from node $i$ to node $j$**.

For this example, the training input and correct target are

$$
(x_1,x_2)=(0,0), \qquad y=0.
$$

## Activation function

Every hidden and output neuron uses the **sigmoid**:

$$
g(s)=\frac{1}{1+e^{-s}}.
$$

Its derivative has a convenient form:

$$
g'(s)=g(s)[1-g(s)]=a(1-a),
$$

where $a=g(s)$ is that neuron's activation.

**Notation:** $\mathrm{in}_j$ is the weighted input to neuron $j$; $a_j=g(\mathrm{in}_j)$ is its output. The symbol $\Delta_j$ below is a *backpropagation error signal*, not a weight change.

# 2. Forward propagation: calculate the prediction

Forward propagation evaluates the neurons **from inputs to output**, using the **current** weights. No weights change during this pass.

## Step 2.1 — Hidden node 3

$$
\begin{aligned}
\mathrm{in}_3
 &= w_{03}x_0+w_{13}x_1+w_{23}x_2\\
 &=0.1(1)+0.1(0)+0.1(0)\\
 &=\boxed{0.1}.
\end{aligned}
$$

Apply sigmoid:

$$
a_3=g(0.1)=\frac{1}{1+e^{-0.1}}
 \approx\boxed{0.52498}.
$$

## Step 2.2 — Hidden node 4

The inputs and initial weights are the same:

$$
\begin{aligned}
\mathrm{in}_4
 &=w_{04}x_0+w_{14}x_1+w_{24}x_2\\
 &=0.1(1)+0.1(0)+0.1(0)=0.1,\\
a_4&=g(0.1)\approx\boxed{0.52498}.
\end{aligned}
$$

**Why are the hidden outputs not zero?** Even when $x_1=x_2=0$, each neuron still receives its **bias input** $x_0=1$, giving a weighted input of $0.1$.

## Step 2.3 — Output node 5

The output neuron receives the **activations** from hidden nodes 3 and 4, not the original inputs directly:

$$
\begin{aligned}
\mathrm{in}_5
 &=w_{05}x_0+w_{35}a_3+w_{45}a_4\\
 &=0.1(1)+0.1(0.52498)+0.1(0.52498)\\
 &\approx\boxed{0.204996}.
\end{aligned}
$$

Then

$$
\hat y=a_5=g(0.204996)\approx\boxed{0.55107}.
$$

**Interpretation:** The desired output is $y=0$, but the network predicts about $0.551$. The weights therefore need adjustment.

# 3. Calculate the loss

For this worked example, use the **half-squared error** for one observation:

$$
\boxed{E=\frac12(y-\hat y)^2}.
$$

With $y=0$ and $\hat y\approx0.55107$:

$$
E=\frac12(0-0.55107)^2\approx\boxed{0.15184}.
$$

The factor $\tfrac12$ is convenient because it cancels the factor $2$ when differentiating. This is the loss convention compatible with the AIMA delta formulas in the uploaded example. Other textbooks may define mean squared error without the factor $\tfrac12$, which changes the gradient's scale.

# 4. Backpropagation: calculate the error signals

The goal is to determine **how each weight affects the loss**. We start at the output and work backward through the network, using the **chain rule**.

## Step 4.1 — Output node 5

The AIMA-style output delta is

$$
\boxed{\Delta_5=g'(\mathrm{in}_5)(y-a_5)}.
$$

First, calculate the sigmoid derivative:

$$
\begin{aligned}
g'(\mathrm{in}_5)
 &=a_5(1-a_5)\\
 &\approx0.55107(1-0.55107)\\
 &\approx0.24739.
\end{aligned}
$$

Now calculate the delta:

$$
\begin{aligned}
\Delta_5
 &=0.24739(0-0.55107)\\
 &\approx\boxed{-0.13634}.
\end{aligned}
$$

**Meaning of the negative sign:** The prediction is higher than the target. With a positive incoming activation, the corresponding output weight should decrease.

## Step 4.2 — Hidden node 3

The hidden-layer delta is

$$
\boxed{\Delta_i=g'(\mathrm{in}_i)\sum_j w_{ij}\Delta_j}.
$$

For node 3, the only downstream neuron is node 5, so the sum has one term:

$$
\begin{aligned}
\Delta_3
 &=a_3(1-a_3)\,w_{35}\Delta_5\\
 &\approx0.52498(1-0.52498)(0.1)(-0.13634)\\
 &\approx\boxed{-0.00340}.
\end{aligned}
$$

## Step 4.3 — Hidden node 4

The network is initially symmetric, so

$$
\Delta_4=a_4(1-a_4)\,w_{45}\Delta_5
\approx\boxed{-0.00340}.
$$

**Important:** Compute all deltas using the **original weights** from the forward pass. Do not update $w_{35}$ before using it to calculate $\Delta_3$.

# 5. Update every weight

The AIMA-style update is

$$
\boxed{w_{ij}^{\mathrm{new}}
 =w_{ij}^{\mathrm{old}}+\alpha a_i\Delta_j}.
$$

For a connection from an original input node, use $a_i=x_i$. For a bias connection, use $a_0=x_0=1$.

This is **gradient descent**, even though the formula has a plus sign: the AIMA delta $\Delta_j$ is defined with the **negative** of the loss-gradient direction. Equivalently, in conventional gradient notation,

$$
\boxed{w^{\mathrm{new}}
 =w^{\mathrm{old}}-\alpha\frac{\partial E}{\partial w}}.
$$

The two formulas are consistent; they use different definitions of the intermediate error signal.

## Step 5.1 — Weights entering output node 5

Using $\alpha=0.1$ and $\Delta_5\approx-0.13634$:

$$
\begin{aligned}
w_{35}^{\mathrm{new}}
 &=0.1+0.1(0.52498)(-0.13634)
 \approx\boxed{0.09284},\\
w_{45}^{\mathrm{new}}
 &=0.1+0.1(0.52498)(-0.13634)
 \approx\boxed{0.09284},\\
w_{05}^{\mathrm{new}}
 &=0.1+0.1(1)(-0.13634)
 \approx\boxed{0.08637}.
\end{aligned}
$$

Using the source document's rounded delta $\Delta_5=-0.1364$ gives $w_{05}=0.08636$; the tiny difference is only from rounding.

## Step 5.2 — Weights entering hidden node 3

Using $\Delta_3\approx-0.00340$:

$$
\begin{aligned}
w_{13}^{\mathrm{new}}
 &=0.1+0.1(0)(-0.00340)=\boxed{0.1},\\
w_{23}^{\mathrm{new}}
 &=0.1+0.1(0)(-0.00340)=\boxed{0.1},\\
w_{03}^{\mathrm{new}}
 &=0.1+0.1(1)(-0.00340)=\boxed{0.09966}.
\end{aligned}
$$

**Why don't $w_{13}$ and $w_{23}$ change?** Because the corresponding inputs are $x_1=x_2=0$. Their weight-update terms are multiplied by zero.

## Step 5.3 — Weights entering hidden node 4

By symmetry,

$$
\begin{aligned}
w_{14}^{\mathrm{new}}&=\boxed{0.1},\\
w_{24}^{\mathrm{new}}&=\boxed{0.1},\\
w_{04}^{\mathrm{new}}&=\boxed{0.09966}.
\end{aligned}
$$

## Summary of updated weights

| Weight | Initial | Updated (approximately) | Reason |
|:--|--:|--:|:--|
| $w_{35}$ | 0.10000 | **0.09284** | Hidden activation and output delta |
| $w_{45}$ | 0.10000 | **0.09284** | Hidden activation and output delta |
| $w_{05}$ | 0.10000 | **0.08636–0.08637** | Bias input equals 1; rounding convention |
| $w_{03}$ | 0.10000 | **0.09966** | Hidden node 3 bias |
| $w_{04}$ | 0.10000 | **0.09966** | Hidden node 4 bias |
| $w_{13},w_{23},w_{14},w_{24}$ | 0.10000 | **0.10000** | Original inputs are both 0 |

# 6. Check whether the update helped

After the weight updates, run **another forward pass** on the *same* input $(0,0)$.

Using the rounded updated weights:

$$
\begin{aligned}
a_3^{\mathrm{new}}&=g(0.09966)\approx0.52489,\\
a_4^{\mathrm{new}}&=g(0.09966)\approx0.52489,\\
\mathrm{in}_5^{\mathrm{new}}
 &\approx0.08636+0.09284(0.52489)+0.09284(0.52489)\\
 &\approx0.18382,\\
\hat y^{\mathrm{new}}
 &=g(0.18382)\approx\boxed{0.54583}.
\end{aligned}
$$

The prediction moved from about **0.55107** toward the desired output **0**. The half-squared error decreases:

$$
E_{\mathrm{before}}\approx0.15184,
\qquad
E_{\mathrm{after}}\approx\frac12(0.54583)^2\approx0.14896.
$$

Thus, **this training step reduced the loss for this one example**. It does not prove the network has learned all four XOR cases.

# 7. Forward propagation vs. backpropagation

| | Forward propagation | Backpropagation |
|:--|:--|:--|
| Direction | Inputs $\rightarrow$ hidden $\rightarrow$ output | Output $\rightarrow$ hidden |
| Main calculation | Weighted sums and activations | Error signals and gradients |
| Main result | Prediction $\hat y$ and loss $E$ | Information needed to update weights |
| Weights | Held fixed during the pass | Updated after deltas are calculated |

**A useful distinction:** *Backpropagation* computes the gradients (or equivalent deltas); the *optimizer*, such as gradient descent, uses those gradients to update the weights. Teaching materials often use “backpropagation” informally for the entire backward-and-update stage.

# 8. Three points students often find confusing

**Why are there two directions?** The prediction requires values flowing from the inputs to the output. To know which earlier weights contributed to the error, the chain rule propagates information backward.

**Why do bias weights change when $x_1=x_2=0$?** The bias input is always $x_0=1$, so its update need not be zero.

**Why is there a plus sign in the AIMA weight update but a minus sign in gradient descent?** AIMA's $\Delta_j$ already incorporates the negative-gradient direction through $(y-a_j)$. The conventional gradient $\partial E/\partial w$ points uphill, so gradient descent subtracts it.

# 9. What comes next to learn XOR?

The XOR targets are:

| $x_1$ | $x_2$ | $y=x_1\oplus x_2$ |
|:--:|:--:|:--:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Training repeats the **forward pass → loss → backward pass → weight update** sequence across these examples, for many epochs. Because XOR is not linearly separable, the hidden layer and its nonlinear activation are important.

**One further caveat:** The source initializes all weights identically. This is convenient for a hand calculation, but symmetric hidden neurons receive identical updates. For effective XOR learning, neural networks usually initialize hidden-neuron weights to **different small random values** to break symmetry.

---

## Source and notation notes

- **Primary numerical source:** Uploaded *ANN Numerical Example x1XORx2 000 (1).pdf*, pp. 1–5. The network, initial values, AIMA deltas, and first-example weight updates are preserved.
- **Lecture context:** Uploaded *ANN(1).pptx*, especially the feed-forward network, multilayer network, and backpropagation sections.
- **Additional teaching explanations:** The loss convention, explicit chain-rule interpretation, gradient-sign clarification, post-update loss check, and symmetry caveat are explanatory additions. The original PDF does not carry out the post-update forward pass.
