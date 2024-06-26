# Neural Network Logic Gate Predictor

This repository contains C codes for neural networks designed to predict the outputs of various logic gates (AND, OR, XOR, etc.). These codes are written from scratch without the use of any external libraries. The primary goal is to demonstrate the fundamentals of neural network implementation and training in C.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Logic Gates Implemented](#logic-gates-implemented)
- [Network Architecture](#network-architecture)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to provide a simple and educational example of how neural networks can be implemented from the ground up in C. The neural networks in this repository are trained to predict the outputs of basic logic gates based on given inputs.

## Getting Started

### Prerequisites

To compile and run the codes in this repository, you need:

- A C compiler (e.g., GCC)
- Basic understanding of C programming
- Basic understanding of neural networks and logic gates

### Installation

Clone this repository to your local machine using the following command:

```sh
git clone https://github.com/Aditya-Codes-247/Neural_Network_C.git
cd NeuralNetworkLogicGates
```

### Logic Gates Implemented

The neural networks in this repository are capable of predicting the outputs of the following logic gates:

- AND
- OR
- XOR

### Network Architecture

The neural networks implemented in this repository consist of:

- An input layer with 2 neurons (corresponding to the 2 inputs of the logic gate)
- One or more hidden layers
- An output layer with 1 neuron (corresponding to the single output of the logic gate)

### Usage

- AND Gate
  ```sh
   gcc -o Neural_AND Neural_AND.c
  ./Neural_AND
  ```
- OR Gate
  ```sh
  gcc -o Neural_OR Neural_OR.c
  ./Neural_OR
  ```
- XOR Gate
  ```sh
  gcc -o Neural_XOR Neural_XOR.c
  ./Neural_XOR
  ```
### Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please create an issue or submit a pull request.

- Steps to Contribute
- Fork this repository
- Create a new branch (git checkout -b feature-branch)
- Commit your changes (git commit -m 'Add some feature')
- Push to the branch (git push origin feature-branch)
- Open a pull request

