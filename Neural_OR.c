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
    return 0.5 * pow(predicted - actual, 2);
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
    const double learning_rate = 0.001;
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double epsilon = 1e-8;
    double hidden_layer[nHiddenNodes];
    double output_layer[nOutNodes];
    double hidden_layer_bias[nHiddenNodes];
    double output_layer_bias[nOutNodes];
    double hidden_weights[nInp][nHiddenNodes];
    double output_weights[nHiddenNodes][nOutNodes];
    double m_output[nHiddenNodes][nOutNodes] = {{0}};
    double v_output[nHiddenNodes][nOutNodes] = {{0}};
    double m_hidden[nInp][nHiddenNodes] = {{0}};
    double v_hidden[nInp][nHiddenNodes] = {{0}};
    double training_inputs[nTrainingSet][nInp] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
    double training_outputs[nTrainingSet][nOutNodes] = {{0.0f}, {1.0f}, {1.0f}, {1.0f}};
    int trainingSetOrder[] = {0, 1, 2, 3};
    int numEpochs = 10000;
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        shuffle(trainingSetOrder, nTrainingSet);
        double total_loss = 0.0;
        for (int x = 0; x < nTrainingSet; x++) {
            int i = trainingSetOrder[x];
            // Forward pass starts here: 
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
            // Backpropagation starts here: 
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
                for (int k = 0; k < nHiddenNodes; k++) {
                    m_output[k][j] = beta1 * m_output[k][j] + (1 - beta1) * deltaOutput[j] * hidden_layer[k];
                    v_output[k][j] = beta2 * v_output[k][j] + (1 - beta2) * deltaOutput[j] * deltaOutput[j];
                    output_weights[k][j] += learning_rate * m_output[k][j] / (sqrt(v_output[k][j]) + epsilon);
                }
                output_layer_bias[j] += learning_rate * deltaOutput[j];
            }
            for (int j = 0; j < nHiddenNodes; j++) {
                for (int k = 0; k < nInp; k++) {
                    m_hidden[k][j] = beta1 * m_hidden[k][j] + (1 - beta1) * deltaHidden[j] * training_inputs[i][k];
                    v_hidden[k][j] = beta2 * v_hidden[k][j] + (1 - beta2) * deltaHidden[j] * deltaHidden[j];
                    hidden_weights[k][j] += learning_rate * m_hidden[k][j] / (sqrt(v_hidden[k][j]) + epsilon);
                }
                hidden_layer_bias[j] += learning_rate * deltaHidden[j];
            }
        }
        printf("Epoch %d, Average Loss: %f\n", epoch + 1, total_loss / nTrainingSet);
    }
    return 0;
}
