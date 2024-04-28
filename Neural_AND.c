#include <stdlib.h>
#include <stdio.h>
#include <math.h>
double relu(double x) {
    return fmax(0, x);
}
double init_wt() {
    return ((double)rand()) / ((double)RAND_MAX);
}
double drelu(double x) {
    return x > 0 ? 1 : 0;
}
double mean_squared_error(double predicted, double actual) {
    return -actual * log(predicted) - (1 - actual) * log(1 - predicted);
}
void shuffle(int *arr, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = arr[j];
            arr[j] = arr[i];
            arr[i] = t;
        }
    }
}
#define nInp 2
#define nHiddenNodes 2
#define nOutNodes 1
#define nTrainingSet 4
int main(void) {
    const double learning_rate = 0.1f;
    double hidden_layer[nHiddenNodes];
    double output_layer[nOutNodes];
    double hidden_layer_bias[nHiddenNodes];
    double output_layer_bias[nOutNodes];
    double hidden_weights[nInp][nHiddenNodes];
    double output_weights[nHiddenNodes][nOutNodes];
    double training_inputs[nTrainingSet][nInp] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
    double training_outputs[nTrainingSet][nOutNodes] = {{0.0f}, {0.0f}, {0.0f}, {1.0f}};
    int trainingSetOrder[] = {0, 1, 2, 3};
    int numEpochs = 10000;
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        shuffle(trainingSetOrder, nTrainingSet);
        double total_loss = 0.0;
        for (int x = 0; x < nTrainingSet; x++) {
            int i = trainingSetOrder[x];
            for (int j = 0; j < nHiddenNodes; j++) {
                double activation = hidden_layer_bias[j];
                for (int k = 0; k < nInp; k++) {
                    activation += training_inputs[i][k] * hidden_weights[k][j];
                }
                hidden_layer[j] = relu(activation);
            }
            for (int j = 0; j < nOutNodes; j++) {
                double activation = output_layer_bias[j];
                for (int k = 0; k < nHiddenNodes; k++) {
                    activation += hidden_layer[k] * output_weights[k][j];
                }
                output_layer[j] = relu(activation);
            }
            double loss = mean_squared_error(output_layer[0], training_outputs[i][0]);
            total_loss += loss;
            printf("Input:%g %g Output:%g    Expected Output: %g\n",
                    training_inputs[i][0], training_inputs[i][1],
                    output_layer[0], training_outputs[i][0]);
            
            double deltaOutput[nOutNodes];
            for (int j = 0; j < nOutNodes; j++) {
                double error = training_outputs[i][j] - output_layer[j];
                deltaOutput[j] = error * drelu(output_layer[j]);
            }
            double deltaHidden[nHiddenNodes];
            for (int j = 0; j < nHiddenNodes; j++) {
                double error = 0.0f;
                for (int k = 0; k < nOutNodes; k++) {
                    error += deltaOutput[k] * output_weights[j][k];
                }
                deltaHidden[j] = error * drelu(hidden_layer[j]);
            }
            for (int j = 0; j < nOutNodes; j++) {
                output_layer_bias[j] += deltaOutput[j] * learning_rate;
                for (int k = 0; k < nHiddenNodes; k++) {
                    output_weights[k][j] += hidden_layer[k] * deltaOutput[j] * learning_rate;
                }
            }
            for (int j = 0; j < nHiddenNodes; j++) {
                hidden_layer_bias[j] += deltaHidden[j] * learning_rate;
                for (int k = 0; k < nInp; k++) {
                    hidden_weights[k][j] += training_inputs[i][k] * deltaHidden[j] * learning_rate;
                }
            }
        }
        printf("Epoch %d, Average Loss: %f\n", epoch + 1, total_loss / nTrainingSet);
    }
    return 0;
}
