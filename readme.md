
# Table of Contents

1.  [Caliberation Gyro](#orgcd1d3df)
    1.  [Introduction](#orgc3ec502)
    2.  [Classical Least Square](#org8b542c8)
2.  [Examples](#org12fda75)
3.  [Instructions](#org4b029a4)
4.  [Progress Bar <code>[3/3]</code>](#org9087410)
5.  [Big Problem other axis](#orgf7122e5)



<a id="orgcd1d3df"></a>

# Caliberation Gyro


<a id="orgc3ec502"></a>

## Introduction

In this repo we&rsquo;ll compare between Caliberation method
The following algorithms are currently supported

-   No Algo
-   Classic Least Square Solution
-   Machine learning methods


<a id="org8b542c8"></a>

## Classical Least Square

Our model of errors is

$$
\tilde{f} = (I_3+S_a+M_a)f+b_a+w_a 
$$

Note the following derivation

$$
\tilde{f}-f = (S_a+M_a)f+b_a
$$

Concating the following matrices would generate the following matrix 

$$
M<sub>3&times; 4</sub>=
\left[

\begin{array}{c | c}
M_a+S_a & b_a
\end{array}

\right]
$$

Given that $ M_{3\times 4} $ matrix we can simplify equation [6](#org3adee00) to be

\\[
~{f} - f<sub>3&times; 1 </sub> = M<sub>3&times; 4</sub>

\begin{pmatrix}
f_{3\times 1 } \\ 1 
\end{pmatrix}

\\]

I&rsquo;ll denote the matrix $ A \in \mathbb{R}^{4\times 6 }$ where $ \omega $ like so 

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

Equation [20](#orgd8a687b) gives us the best $M$ matrix for a given measurement
so our estimator is an **Unbiased estimator**, so for simple calculation we&rsquo;ll have

$$
\hat{M}=\frac{1}{n}\sum_{i=1}^{n}M = \frac{1}{n}\sum_{i=1}^{n}z_{3\times 6}A^{T}(AA^{T})^{-1}
$$

The Classical Caliberation method uses the formula from the lectures


<a id="org12fda75"></a>

# Examples

We want something like this photo
![img](./example1.png)


<a id="org4b029a4"></a>

# Instructions

For running the algorithms

    python automate.py

The same result can be achieved even easily with 

    python simple_api.py


<a id="org9087410"></a>

# Progress Bar <code>[3/3]</code>

Here are the Agenda to do 

-   [X] No Algo
-   [X] Classical Least Square
-   [X] Machine learning method


<a id="orgf7122e5"></a>

# Big Problem other axis

