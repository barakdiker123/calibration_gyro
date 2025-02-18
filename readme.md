
# Table of Contents

1.  [Caliberation Gyro](#org2b408a3)
    1.  [Introduction](#org1465cd8)
    2.  [Classical Least Square](#org49ea488)
2.  [Examples](#org35a0e55)
3.  [Instructions](#orga236232)
4.  [Progress Bar <code>[3/3]</code>](#org198f6ae)
5.  [Big Problem other axis](#org5caa8db)



<a id="org2b408a3"></a>

# Caliberation Gyro


<a id="org1465cd8"></a>

## Introduction

In this repo we&rsquo;ll compare between Caliberation method
The following algorithms are currently supported

-   No Algo
-   Classic Least Square Solution
-   Machine learning methods


<a id="org49ea488"></a>

## Classical Least Square

Our model of errors is

$$
\tilde{f} = (I_3+S_a+M_a)f+b_a+w_a 
$$

Note the following derivation

$$
\tilde{f}-f = (S_a+M_a)f+b_a
$$

I&rsquo;ll denote the matrix $A \in \mathbb{R}^{4\times 6 }$ where $\omega$ like so 

$$
A_{4\times 6}=\begin{pmatrix}
-\omega & \omega  &  0 & 0 & 0 & 0 \\ 
0 & 0  &  -\omega & \omega & 0  & 0 \\ 
0 & 0  &  0 & 0 & -\omega  & \omega \\ 
1 & 1 &  1 & 1 & 1 &1 \\ 
\end{pmatrix}
$$

and

$$
z_{3\times 6}=M_{3\times 4 }A_{4\times 6}
$$

$$
z_{3\times 6}A^{T}(AA^{T})^{-1}=M_{3\times 4 }A_{4\times 6}A^{T}(AA^{T})^{-1}
$$

$$
z_{3\times 6}A^{T}(AA^{T})^{-1}=M_{3\times 4 }
$$

Equation [14](#org9a3bc0e) gives us the best $M$ matrix for a given measurement
so our estimator is an **Unbiased estimator**, so for simple calculation we&rsquo;ll have

$$
\hat{M}=\frac{1}{n}\sum_{i=1}^{n}M = \frac{1}{n}\sum_{i=1}^{n}z_{3\times 6}A^{T}(AA^{T})^{-1}
$$

The Classical Caliberation method uses the formula from the lectures


<a id="org35a0e55"></a>

# Examples

We want something like this photo
![img](./example1.png)


<a id="orga236232"></a>

# Instructions

For running the algorithms

    python automate.py

The same result can be achieved even easily with 

    python simple_api.py


<a id="org198f6ae"></a>

# Progress Bar <code>[3/3]</code>

Here are the Agenda to do 

-   [X] No Algo
-   [X] Classical Least Square
-   [X] Machine learning method


<a id="org5caa8db"></a>

# Big Problem other axis

