# Training a Perceptron for an AND Gate

**Reference:** zyBooks, Activity 12.2.9 — *Training a perceptron model for an AND gate*

## 1. The AND gate and its training data

An AND gate outputs 1 only when **both** inputs are 1.

| Input $x_1$ | Input $x_2$ | Actual output $y$ |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 1 | 1 |

A perceptron computes a **weighted sum**:

$$
z = w_1x_1 + w_2x_2 + b
$$

It then predicts:

$$
\hat y =
\begin{cases}
1, & z \geq 0,\\
0, & z < 0.
\end{cases}
$$

Thus, a point **on** the decision boundary is classified as 1.

## 2. Initial weights and decision boundary

The activity initializes the perceptron with:

$$
w_1 = 1, \qquad w_2 = 1, \qquad b = -1.
$$

The decision boundary consists of points for which $z=0$:

$$
w_1x_1 + w_2x_2 + b = 0.
$$

Substitute the initial parameters:

$$
x_1 + x_2 - 1 = 0.
$$

Solve for $x_2$:

$$
\boxed{x_2 = 1 - x_1}
$$

This line passes through $(0,1)$ and $(1,0)$. Both points lie **on** the initial boundary and would be predicted as class 1 if evaluated before any updates. However, the activity processes training points **sequentially**, updating the parameters immediately after a mistake.

## 3. Process the first training point: $(0,0)$

Calculate the weighted sum:

$$
z = 1(0) + 1(0) - 1 = -1.
$$

Since $z<0$, the prediction is $\hat y=0$. The actual AND output is $y=0$, so the prediction is correct. **No update is needed.**

## 4. Process $(1,0)$: an incorrect prediction

Using the initial parameters:

$$
z = 1(1) + 1(0) - 1 = 0.
$$

Since $z\geq0$, the perceptron predicts $\hat y=1$. But the actual output is $y=0$.

The error is:

$$
e = y - \hat y = 0 - 1 = \boxed{-1}.
$$

The perceptron must update its parameters.

## 5. The perceptron learning rule

For each weight $w_i$, the update rule is:

$$
\boxed{w_i^{\text{new}} = w_i^{\text{old}} + r(y-\hat y)x_i}
$$

The bias is updated separately:

$$
\boxed{b^{\text{new}} = b^{\text{old}} + r(y-\hat y)}
$$

where:

- $r$ is the **learning rate**; here, $r=0.1$.
- $y$ is the **actual** class.
- $\hat y$ is the **predicted** class.
- $x_i$ is the input associated with weight $w_i$.

For the incorrectly classified point $(x_1,x_2)=(1,0)$, we have:

$$
r=0.1,\quad y-\hat y=-1,\quad x_1=1,\quad x_2=0.
$$

### 5.1 Update $w_1$

$$
\begin{aligned}
w_1^{\text{new}}
&= w_1^{\text{old}} + r(y-\hat y)x_1\\
&= 1 + 0.1(-1)(1)\\
&= 1 - 0.1\\
&= \boxed{0.9}.
\end{aligned}
$$

**Why did $w_1$ change?** Because $x_1=1$, the error contributes a nonzero adjustment of $-0.1$.

### 5.2 Update $w_2$

$$
\begin{aligned}
w_2^{\text{new}}
&= w_2^{\text{old}} + r(y-\hat y)x_2\\
&= 1 + 0.1(-1)(0)\\
&= 1 + 0\\
&= \boxed{1}.
\end{aligned}
$$

**Why did $w_2$ stay the same?** Because $x_2=0$, multiplying the error by this input gives zero. No adjustment is applied to $w_2$.

### 5.3 Update the bias $b$

$$
\begin{aligned}
b^{\text{new}}
&= b^{\text{old}} + r(y-\hat y)\\
&= -1 + 0.1(-1)\\
&= \boxed{-1.1}.
\end{aligned}
$$

Unlike the weights, the bias update does **not** multiply by $x_1$ or $x_2$.

### 5.4 Summary of the parameter updates

| Parameter | Before | Adjustment | After |
|---|---:|---:|---:|
| $w_1$ | 1 | $0.1(-1)(1)=-0.1$ | **0.9** |
| $w_2$ | 1 | $0.1(-1)(0)=0$ | **1** |
| $b$ | -1 | $0.1(-1)=-0.1$ | **-1.1** |

## 6. Derive the new decision boundary

With the updated parameters:

$$
w_1=0.9,\qquad w_2=1,\qquad b=-1.1.
$$

Set the weighted sum equal to zero:

$$
0.9x_1 + x_2 - 1.1 = 0.
$$

Rearrange:

$$
\boxed{x_2 = 1.1 - 0.9x_1}
$$

Compare the two lines:

| Property | Initial boundary | Updated boundary |
|---|---|---|
| Equation | $x_2=1-x_1$ | $x_2=1.1-0.9x_1$ |
| Slope | $-1$ | $-0.9$ |
| Vertical intercept | $1$ | $1.1$ |

The line moves and rotates slightly because the **weight and bias values changed**.

## 7. Verify the predictions after the update

Use the updated weighted sum:

$$
z = 0.9x_1 + x_2 - 1.1.
$$

| Point $(x_1,x_2)$ | Weighted sum $z$ | Predicted $\hat y$ | Actual $y$ | Correct? |
|---|---|:---:|:---:|:---:|
| $(0,0)$ | $0.9(0)+0-1.1=-1.1$ | 0 | 0 | Yes |
| $(1,0)$ | $0.9(1)+0-1.1=-0.2$ | 0 | 0 | Yes |
| $(0,1)$ | $0.9(0)+1-1.1=-0.1$ | 0 | 0 | Yes |
| $(1,1)$ | $0.9(1)+1-1.1=0.8$ | 1 | 1 | Yes |

All four AND-gate inputs are now classified correctly.

**Important:** The point $(0,1)$ was on the **initial** boundary, but it is evaluated **after** the update triggered by $(1,0)$. Under the updated parameters, its weighted sum is $-0.1$, so it is correctly classified as 0.

## 8. What happens in Epoch 2?

An **epoch** is one complete pass through the training examples.

In Epoch 2, the perceptron evaluates all four points using the updated parameters. Because every prediction is already correct, the error for each point is zero:

$$
y-\hat y=0.
$$

Therefore, all weight and bias adjustments are zero. **The decision boundary does not change between the end of Epoch 1 and Epoch 2.**

## 9. Main takeaways

1. The perceptron updates parameters **when it makes a mistake**.
2. Each weight changes in proportion to its corresponding input: $\Delta w_i=r(y-\hat y)x_i$.
3. For $(1,0)$, $w_1$ changes but $w_2$ does not, because $x_2=0$.
4. The bias changes according to $\Delta b=r(y-\hat y)$.
5. Updating the weights and bias changes the decision boundary.
6. Once all AND-gate inputs are classified correctly, the next epoch requires no further updates.

---

*Formatting note:* This file uses standard Markdown with dollar-delimited LaTeX math ($...$ and $$...$$). GitHub and compatible Markdown previews render the equations. zyBooks Brief notes may not support this math syntax directly.
