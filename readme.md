
# Table of Contents

1.  [Caliberation Gyro](#org168705f)
    1.  [Introduction](#orga60e887)
    2.  [Classical Least Square](#org260bc9c)
2.  [Examples](#org979f19d)
3.  [Instructions](#org1c3fe14)
4.  [Progress Bar <code>[3/3]</code>](#orgc726241)
5.  [Big Problem other axis](#org5854f65)



<a id="org168705f"></a>

# Caliberation Gyro


<a id="orga60e887"></a>

## Introduction

In this repo we&rsquo;ll compare between Caliberation method
The following algorithms are currently supported

-   No Algo
-   Classic Least Square Solution
-   Machine learning methods


<a id="org260bc9c"></a>

## Classical Least Square

Our model of errors is
\label{equ:1}

\begin{equation}
\tilde{f} = (I_3+S_a+M_a)f+b_a+w_a 
\end{equation}

Where

\begin{math}
S_a\in diag_{3\times 3}\mathbb{R}
\end{math}

and 

\begin{math}
M_a\in AntiSymm_{3\times 3}\mathbb{R}
\end{math}

and

\begin{math}
b_a\in \mathbb{R}^3
\end{math}

Note the following derivation

\begin{equation*}
\tilde{f}-f = (S_a+M_a)f+b_a
\end{equation*}

Concating the following matrices would generate the following matrix 

\begin{equation*}
M_{3\times 4}=
\left[
\begin{array}{c | c}
M_a+S_a & b_a
\end{array}
\right]
\end{equation*}

Given that $M_{3\times 4}$ matrix we can simplify equation \ref{equ:1}
to be

\begin{equation*}
\tilde{f} - f_{3\times 1 } = M_{3\times 4}
\begin{pmatrix}
f_{3\times 1 } \\ 1 
\end{pmatrix}
\end{equation*}

For example, Estimating left column of M with down x direction measurements

\begin{equation*}
\tilde{f}_{down}^x-
\begin{pmatrix}
-g \\ 0 \\ 0 
\end{pmatrix} = M
\begin{pmatrix}
-g \\ 0 \\ 0 \\ 1
\end{pmatrix}
\end{equation*}

We can concate all the vector equations together and we&rsquo;ll infer the following matrix equation

\begin{equation*}
\begin{pmatrix}
\tilde{f}_{down}^{x} & \tilde{f}_{x}^{up} & \cdots & \tilde{f}_{up}^{z} 
\end{pmatrix}
-
\begin{pmatrix}
-g & g  & 0 & 0 & 0 &   0 \\
0 & 0 & -g & g & 0  & 0 \\
0 & 0 & 0 & 0  & -g & g
\end{pmatrix}
=M
\begin{pmatrix}
-g & g  &  0 & 0 & 0 & 0 \\ 
0 & 0  &  -g & g & 0  & 0 \\ 
0 & 0  &  0 & 0 & -g  & g \\ 
1 & 1 &  1 & 1 & 1 &1 \\ 
\end{pmatrix}
\end{equation*}

I&rsquo;ll denote the matrix $A \in \mathbb{R}^{4\times 6 }$ like so 

\begin{equation*}
A_{4\times 6}=\begin{pmatrix}
-g & g  &  0 & 0 & 0 & 0 \\ 
0 & 0  &  -g & g & 0  & 0 \\ 
0 & 0  &  0 & 0 & -g  & g \\ 
1 & 1 &  1 & 1 & 1 &1 \\ 
\end{pmatrix}
\end{equation*}

and

\begin{equation*}
z_{3\times 6}=M_{3\times 4 }A_{4\times 6}
\end{equation*}

\begin{equation*}
z_{3\times 6}A^{T}(AA^{T})^{-1}=M_{3\times 4 }A_{4\times 6}A^{T}(AA^{T})^{-1}
\end{equation*}

\begin{equation}\label{eq:LS}
z_{3\times 6}A^{T}(AA^{T})^{-1}=M_{3\times 4 }
\end{equation}

Equation \ref{eq:LS} gives us the best $M$ matrix for a given measurement
so our estimator is an \bold{Unbiased estimator}, so for simple calculation we&rsquo;ll have

\begin{equation}\label{eq:LS2}
\hat{M}=\frac{1}{n}\sum_{i=1}^{n}M = \frac{1}{n}\sum_{i=1}^{n}z_{3\times 6}A^{T}(AA^{T})^{-1}
\end{equation}

The Classical Caliberation method uses the formula from the lectures


<a id="org979f19d"></a>

# Examples

We want something like this photo
![img](./example1.png)


<a id="org1c3fe14"></a>

# Instructions

For running the algorithms

    python automate.py

The same result can be achieved even easily with 

    python simple_api.py


<a id="orgc726241"></a>

# Progress Bar <code>[3/3]</code>

Here are the Agenda to do 

-   [X] No Algo
-   [X] Classical Least Square
-   [X] Machine learning method


<a id="org5854f65"></a>

# Big Problem other axis

