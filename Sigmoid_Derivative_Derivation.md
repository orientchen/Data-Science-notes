# Sigmoid Derivative

You have the sigmoid function:

$$
g(z)=\frac{1}{1+e^{-z}}
$$

We want $g'(z)$. Let's go step by step.

## Step 1: Rewrite $g(z)$

$$
g(z)=(1+e^{-z})^{-1}
$$

## Step 2: Differentiate using the chain rule

$$
g'(z)=-1\cdot(1+e^{-z})^{-2}\cdot\frac{d}{dz}(1+e^{-z})
$$

$$
\frac{d}{dz}(1+e^{-z})=-e^{-z}
$$

So,

$$
g'(z)=-(1+e^{-z})^{-2}\cdot(-e^{-z})=\frac{e^{-z}}{(1+e^{-z})^2}
$$

## Step 3: Simplify in terms of $g(z)$

Notice that:

$$
g(z)=\frac{1}{1+e^{-z}}\implies 1-g(z)=\frac{e^{-z}}{1+e^{-z}}
$$

Then,

$$
g'(z)=\frac{e^{-z}}{(1+e^{-z})^2}
=\left(\frac{1}{1+e^{-z}}\right)
\left(\frac{e^{-z}}{1+e^{-z}}\right)
=g(z)(1-g(z))
$$
