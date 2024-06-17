#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 28
#define FILTER_SIZE 3
#define PADDING 1
#define INPUT_SIZE_PADDED (INPUT_SIZE + 2 * PADDING)
#define OUTPUT_SIZE_PADDED (INPUT_SIZE - FILTER_SIZE + 1 + 2 * PADDING)
#define POOL_SIZE 2
#define POOL_OUTPUT_SIZE (OUTPUT_SIZE_PADDED / POOL_SIZE)
#define NUM_FILTERS 8
#define FC_INPUT_SIZE (NUM_FILTERS * POOL_OUTPUT_SIZE * POOL_OUTPUT_SIZE)
#define FC_OUTPUT_SIZE 10
#define NUM_TRAIN 60000
#define NUM_TEST 10000

typedef struct {
    float* m;
    float* v;
    float beta1;
    float beta2;
    float epsilon;
    int t;
} AdamOptimizer;

uint32_t reverse_int(uint32_t i) {
    uint8_t c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

void read_mnist_images(const char *filename, float images[][INPUT_SIZE][INPUT_SIZE], int num_images) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Cannot open file");
        exit(1);
    }

    uint32_t magic_number = 0;
    uint32_t number_of_images = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;

    fread(&magic_number, sizeof(magic_number), 1, file);
    fread(&number_of_images, sizeof(number_of_images), 1, file);
    fread(&rows, sizeof(rows), 1, file);
    fread(&cols, sizeof(cols), 1, file);

    magic_number = reverse_int(magic_number);
    number_of_images = reverse_int(number_of_images);
    rows = reverse_int(rows);
    cols = reverse_int(cols);

    if (rows != INPUT_SIZE || cols != INPUT_SIZE) {
        fprintf(stderr, "Invalid image dimensions: %d x %d\n", rows, cols);
        exit(1);
    }

    for (int i = 0; i < num_images; i++) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                uint8_t pixel = 0;
                fread(&pixel, sizeof(pixel), 1, file);
                images[i][r][c] = pixel / 255.0;
            }
        }
    }

    fclose(file);
}

void read_mnist_labels(const char *filename, float labels[][FC_OUTPUT_SIZE], int num_labels) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Cannot open file");
        exit(1);
    }

    uint32_t magic_number = 0;
    uint32_t number_of_labels = 0;

    fread(&magic_number, sizeof(magic_number), 1, file);
    fread(&number_of_labels, sizeof(number_of_labels), 1, file);

    magic_number = reverse_int(magic_number);
    number_of_labels = reverse_int(number_of_labels);

    for (int i = 0; i < num_labels; i++) {
        uint8_t label = 0;
        fread(&label, sizeof(label), 1, file);
        for (int j = 0; j < FC_OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }

    fclose(file);
}

void pad_input(float input[INPUT_SIZE][INPUT_SIZE], float padded_input[INPUT_SIZE_PADDED][INPUT_SIZE_PADDED]) {
    for (int i = 0; i < INPUT_SIZE_PADDED; i++) {
        for (int j = 0; j < INPUT_SIZE_PADDED; j++) {
            if (i < PADDING || i >= INPUT_SIZE_PADDED - PADDING || j < PADDING || j >= INPUT_SIZE_PADDED - PADDING) {
                padded_input[i][j] = 0;
            } else {
                padded_input[i][j] = input[i - PADDING][j - PADDING];
            }
        }
    }
}

void convolution(float input[INPUT_SIZE_PADDED][INPUT_SIZE_PADDED], float filter[FILTER_SIZE][FILTER_SIZE], float bias, float output[OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED]) {
    for (int i = 0; i < OUTPUT_SIZE_PADDED; i++) {
        for (int j = 0; j < OUTPUT_SIZE_PADDED; j++) {
            float sum = 0.0;
            for (int k = 0; k < FILTER_SIZE; k++) {
                for (int l = 0; l < FILTER_SIZE; l++) {
                    sum += input[i + k][j + l] * filter[k][l];
                }
            }
            output[i][j] = sum + bias;
        }
    }
}

void convolution_layer(float input[INPUT_SIZE_PADDED][INPUT_SIZE_PADDED], float filters[NUM_FILTERS][FILTER_SIZE][FILTER_SIZE], float biases[NUM_FILTERS], float output[NUM_FILTERS][OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED]) {
    for (int f = 0; f < NUM_FILTERS; f++) {
        convolution(input, filters[f], biases[f], output[f]);
    }
}

void relu(float input[NUM_FILTERS][OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED]) {
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int i = 0; i < OUTPUT_SIZE_PADDED; i++) {
            for (int j = 0; j < OUTPUT_SIZE_PADDED; j++) {
                if (input[f][i][j] < 0) {
                    input[f][i][j] = 0;
                }
            }
        }
    }
}

void max_pooling(float input[OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED], float output[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE]) {
    for (int i = 0; i < POOL_OUTPUT_SIZE; i++) {
        for (int j = 0; j < POOL_OUTPUT_SIZE; j++) {
            float max = input[i * POOL_SIZE][j * POOL_SIZE];
            for (int k = 0; k < POOL_SIZE; k++) {
                for (int l = 0; l < POOL_SIZE; l++) {
                    if (input[i * POOL_SIZE + k][j * POOL_SIZE + l] > max) {
                        max = input[i * POOL_SIZE + k][j * POOL_SIZE + l];
                    }
                }
            }
            output[i][j] = max;
        }
    }
}

void max_pooling_layer(float input[NUM_FILTERS][OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED], float output[NUM_FILTERS][POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE]) {
    for (int f = 0; f < NUM_FILTERS; f++) {
        max_pooling(input[f], output[f]);
    }
}

void flatten(float input[NUM_FILTERS][POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE], float output[FC_INPUT_SIZE]) {
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int i = 0; i < POOL_OUTPUT_SIZE; i++) {
            for (int j = 0; j < POOL_OUTPUT_SIZE; j++) {
                output[f * POOL_OUTPUT_SIZE * POOL_OUTPUT_SIZE + i * POOL_OUTPUT_SIZE + j] = input[f][i][j];
            }
        }
    }
}

void fully_connected(float input[FC_INPUT_SIZE], float weights[FC_OUTPUT_SIZE][FC_INPUT_SIZE], float biases[FC_OUTPUT_SIZE], float output[FC_OUTPUT_SIZE]) {
    for (int i = 0; i < FC_OUTPUT_SIZE; i++) {
        output[i] = biases[i];
        for (int j = 0; j < FC_INPUT_SIZE; j++) {
            output[i] += input[j] * weights[i][j];
        }
    }
}

void initialize_weights_xavier(float weights[FC_OUTPUT_SIZE][FC_INPUT_SIZE], float biases[FC_OUTPUT_SIZE]) {
    srand(time(NULL));
    for (int i = 0; i < FC_OUTPUT_SIZE; i++) {
        biases[i] = 0.0;
        for (int j = 0; j < FC_INPUT_SIZE; j++) {
            weights[i][j] = ((float)rand() / RAND_MAX - 0.5) * sqrt(2.0 / FC_INPUT_SIZE);
        }
    }
}

void initialize_filters_xavier(float filters[NUM_FILTERS][FILTER_SIZE][FILTER_SIZE], float biases[NUM_FILTERS]) {
    srand(time(NULL));
    for (int i = 0; i < NUM_FILTERS; i++) {
        biases[i] = 0.0;
        for (int j = 0; j < FILTER_SIZE; j++) {
            for (int k = 0; k < FILTER_SIZE; k++) {
                filters[i][j][k] = ((float)rand() / RAND_MAX - 0.5) * sqrt(2.0 / (FILTER_SIZE * FILTER_SIZE));
            }
        }
    }
}

void softmax(float* input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        input[i] = exp(input[i] - max_val);
        sum += input[i];
    }

    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

float cross_entropy_loss(float* predicted, float* actual, int size) {
    float loss = 0.0;
    for (int i = 0; i < size; i++) {
        if (predicted[i] >= 0) {
            loss -= actual[i] * log(predicted[i] + 1e-9);
        } else {
            printf("Warning: predicted[%d] = %f is non-positive\n", i, predicted[i]);
        }
    }
    return loss / size;
}

float accuracy(float* predicted, float* actual, int size) {
    int correct = 0;
    for (int i = 0; i < size; i++) {
        if ((predicted[i] >= 0.5 && actual[i] == 1.0) || (predicted[i] < 0.5 && actual[i] == 0.0)) {
            correct++;
        }
    }
    return (float)correct / size;
}

void update_weights(float* weights, float* dL_dweights, int size, float learning_rate) {
    for (int i = 0; i < size; i++) {
        weights[i] -= learning_rate * dL_dweights[i];
    }
}

void update_biases(float* biases, float* dL_dbiases, int size, float learning_rate) {
    for (int i = 0; i < size; i++) {
        biases[i] -= learning_rate * dL_dbiases[i];
    }
}

void fully_connected_backward(float* input, float* weights, float* biases, float* dL_dout, float* dL_din, float* dL_dweights, float* dL_dbiases, int input_size, int output_size) {
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            dL_dweights[i * input_size + j] = dL_dout[i] * input[j];
        }
        dL_dbiases[i] = dL_dout[i];
    }

    for (int j = 0; j < input_size; j++) {
        dL_din[j] = 0;
        for (int i = 0; i < output_size; i++) {
            dL_din[j] += weights[i * input_size + j] * dL_dout[i];
        }
    }
}

void max_pooling_backward(float input[NUM_FILTERS][OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED], float grad_output[NUM_FILTERS][POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE], float grad_input[NUM_FILTERS][OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED]) {
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int i = 0; i < OUTPUT_SIZE_PADDED; i++) {
            for (int j = 0; j < OUTPUT_SIZE_PADDED; j++) {
                grad_input[f][i][j] = 0;
            }
        }

        for (int i = 0; i < POOL_OUTPUT_SIZE; i++) {
            for (int j = 0; j < POOL_OUTPUT_SIZE; j++) {
                int max_i = i * POOL_SIZE;
                int max_j = j * POOL_SIZE;
                for (int k = 0; k < POOL_SIZE; k++) {
                    for (int l = 0; l < POOL_SIZE; l++) {
                        if (input[f][i * POOL_SIZE + k][j * POOL_SIZE + l] > input[f][max_i][max_j]) {
                            max_i = i * POOL_SIZE + k;
                            max_j = j * POOL_SIZE + l;
                        }
                    }
                }
                grad_input[f][max_i][max_j] = grad_output[f][i][j];
            }
        }
    }
}

void relu_backward(float input[NUM_FILTERS][OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED], float grad_output[NUM_FILTERS][OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED], float grad_input[NUM_FILTERS][OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED]) {
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int i = 0; i < OUTPUT_SIZE_PADDED; i++) {
            for (int j = 0; j < OUTPUT_SIZE_PADDED; j++) {
                grad_input[f][i][j] = (input[f][i][j] > 0) ? grad_output[f][i][j] : 0;
            }
        }
    }
}

void convolution_backward(float input[INPUT_SIZE_PADDED][INPUT_SIZE_PADDED], float filter[FILTER_SIZE][FILTER_SIZE], float grad_output[OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED], float grad_input[INPUT_SIZE_PADDED][INPUT_SIZE_PADDED], float grad_filter[FILTER_SIZE][FILTER_SIZE], float* grad_bias) {
    for (int i = 0; i < INPUT_SIZE_PADDED; i++) {
        for (int j = 0; j < INPUT_SIZE_PADDED; j++) {
            grad_input[i][j] = 0;
        }
    }

    for (int i = 0; i < FILTER_SIZE; i++) {
        for (int j = 0; j < FILTER_SIZE; j++) {
            grad_filter[i][j] = 0;
        }
    }

    *grad_bias = 0;

    for (int i = 0; i < OUTPUT_SIZE_PADDED; i++) {
        for (int j = 0; j < OUTPUT_SIZE_PADDED; j++) {
            *grad_bias += grad_output[i][j];
            for (int k = 0; k < FILTER_SIZE; k++) {
                for (int l = 0; l < FILTER_SIZE; l++) {
                    grad_filter[k][l] += input[i + k][j + l] * grad_output[i][j];
                    grad_input[i + k][j + l] += filter[k][l] * grad_output[i][j];
                }
            }
        }
    }
}

void forward_pass(float input[INPUT_SIZE_PADDED][INPUT_SIZE_PADDED], float filters[NUM_FILTERS][FILTER_SIZE][FILTER_SIZE], float conv_biases[NUM_FILTERS], float conv_output[NUM_FILTERS][OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED], float pool_output[NUM_FILTERS][POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE], float fc_input[FC_INPUT_SIZE], float fc_weights[FC_OUTPUT_SIZE][FC_INPUT_SIZE], float fc_biases[FC_OUTPUT_SIZE], float fc_output[FC_OUTPUT_SIZE]) {
    convolution_layer(input, filters, conv_biases, conv_output);
    relu(conv_output);
    max_pooling_layer(conv_output, pool_output);
    flatten(pool_output, fc_input);
    fully_connected(fc_input, fc_weights, fc_biases, fc_output);
    softmax(fc_output, FC_OUTPUT_SIZE);
}

void backward_pass(float input[INPUT_SIZE_PADDED][INPUT_SIZE_PADDED], float filters[NUM_FILTERS][FILTER_SIZE][FILTER_SIZE], float conv_biases[NUM_FILTERS], float conv_output[NUM_FILTERS][OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED], float pool_output[NUM_FILTERS][POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE], float fc_input[FC_INPUT_SIZE], float fc_weights[FC_OUTPUT_SIZE][FC_INPUT_SIZE], float fc_biases[FC_OUTPUT_SIZE], float fc_output[FC_OUTPUT_SIZE], float target[FC_OUTPUT_SIZE], float dL_dout[FC_OUTPUT_SIZE], float dL_din[FC_INPUT_SIZE], float dL_dweights[FC_OUTPUT_SIZE][FC_INPUT_SIZE], float dL_dbiases[FC_OUTPUT_SIZE], float grad_relu[NUM_FILTERS][OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED], float grad_pool[NUM_FILTERS][OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED], float grad_conv[NUM_FILTERS][INPUT_SIZE_PADDED][INPUT_SIZE_PADDED], float grad_filters[NUM_FILTERS][FILTER_SIZE][FILTER_SIZE], float grad_biases[NUM_FILTERS]) {
    for (int i = 0; i < FC_OUTPUT_SIZE; i++) {
        dL_dout[i] = fc_output[i] - target[i];
    }
    fully_connected_backward(fc_input, (float*)fc_weights, fc_biases, dL_dout, dL_din, (float*)dL_dweights, dL_dbiases, FC_INPUT_SIZE, FC_OUTPUT_SIZE);


    max_pooling_backward(conv_output, (float(*)[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE])dL_din, grad_pool);


    relu_backward(conv_output, grad_pool, grad_relu);

    for (int f = 0; f < NUM_FILTERS; f++) {
        convolution_backward(input, filters[f], grad_relu[f], grad_conv[f], grad_filters[f], &grad_biases[f]);
    }
}

void initialize_adam(AdamOptimizer* optimizer, int size) {
    optimizer->m = (float*)calloc(size, sizeof(float));
    optimizer->v = (float*)calloc(size, sizeof(float));
    optimizer->beta1 = 0.9;
    optimizer->beta2 = 0.999;
    optimizer->epsilon = 1e-8;
    optimizer->t = 0;
}

void update_adam(AdamOptimizer* optimizer, float* param, float* grad, int size, float learning_rate) {
    optimizer->t += 1;
    for (int i = 0; i < size; i++) {
        optimizer->m[i] = optimizer->beta1 * optimizer->m[i] + (1 - optimizer->beta1) * grad[i];
        optimizer->v[i] = optimizer->beta2 * optimizer->v[i] + (1 - optimizer->beta2) * grad[i] * grad[i];

        float m_hat = optimizer->m[i] / (1 - pow(optimizer->beta1, optimizer->t));
        float v_hat = optimizer->v[i] / (1 - pow(optimizer->beta2, optimizer->t));

        param[i] -= learning_rate * m_hat / (sqrt(v_hat) + optimizer->epsilon);
    }
}

void free_adam(AdamOptimizer* optimizer) {
    free(optimizer->m);
    free(optimizer->v);
}

void train(float inputs[][INPUT_SIZE][INPUT_SIZE], float targets[][FC_OUTPUT_SIZE], int num_samples, int epochs, float learning_rate) {
    float padded_input[INPUT_SIZE_PADDED][INPUT_SIZE_PADDED];
    float filters[NUM_FILTERS][FILTER_SIZE][FILTER_SIZE];
    float conv_biases[NUM_FILTERS];
    float conv_output[NUM_FILTERS][OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED];
    float pool_output[NUM_FILTERS][POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE];
    float fc_input[FC_INPUT_SIZE];
    float fc_weights[FC_OUTPUT_SIZE][FC_INPUT_SIZE];
    float fc_biases[FC_OUTPUT_SIZE];
    float fc_output[FC_OUTPUT_SIZE];

    float dL_dout[FC_OUTPUT_SIZE];
    float dL_din[FC_INPUT_SIZE];
    float dL_dweights[FC_OUTPUT_SIZE][FC_INPUT_SIZE];
    float dL_dbiases[FC_OUTPUT_SIZE];
    float grad_relu[NUM_FILTERS][OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED];
    float grad_pool[NUM_FILTERS][OUTPUT_SIZE_PADDED][OUTPUT_SIZE_PADDED];
    float grad_conv[NUM_FILTERS][INPUT_SIZE_PADDED][INPUT_SIZE_PADDED];
    float grad_filters[NUM_FILTERS][FILTER_SIZE][FILTER_SIZE];
    float grad_biases[NUM_FILTERS];

    initialize_filters_xavier(filters, conv_biases);
    initialize_weights_xavier(fc_weights, fc_biases);

    AdamOptimizer adam_fc_weights;
    initialize_adam(&adam_fc_weights, FC_OUTPUT_SIZE * FC_INPUT_SIZE);

    AdamOptimizer adam_fc_biases;
    initialize_adam(&adam_fc_biases, FC_OUTPUT_SIZE);

    AdamOptimizer adam_filters[NUM_FILTERS];
    for (int i = 0; i < NUM_FILTERS; i++) {
        initialize_adam(&adam_filters[i], FILTER_SIZE * FILTER_SIZE);
    }

    AdamOptimizer adam_conv_biases;
    initialize_adam(&adam_conv_biases, NUM_FILTERS);

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0;
        float epoch_accuracy = 0.0;
        for (int sample = 0; sample < num_samples; sample++) {
            float (*input)[INPUT_SIZE] = inputs[sample];
            float (*target) = targets[sample];

            pad_input(input, padded_input);

            forward_pass(padded_input, filters, conv_biases, conv_output, pool_output, fc_input, fc_weights, fc_biases, fc_output);

            float loss = cross_entropy_loss(fc_output, target, FC_OUTPUT_SIZE);
            epoch_loss += loss;

            float acc = accuracy(fc_output, target, FC_OUTPUT_SIZE);
            epoch_accuracy += acc;

            backward_pass(padded_input, filters, conv_biases, conv_output, pool_output, fc_input, fc_weights, fc_biases, fc_output, target, dL_dout, dL_din, dL_dweights, dL_dbiases, grad_relu, grad_pool, grad_conv, grad_filters, grad_biases);

            update_adam(&adam_fc_weights, (float*)fc_weights, (float*)dL_dweights, FC_OUTPUT_SIZE * FC_INPUT_SIZE, learning_rate);
            update_adam(&adam_fc_biases, fc_biases, dL_dbiases, FC_OUTPUT_SIZE, learning_rate);

            for (int f = 0; f < NUM_FILTERS; f++) {
                update_adam(&adam_filters[f], (float*)filters[f], (float*)grad_filters[f], FILTER_SIZE * FILTER_SIZE, learning_rate);
            }
            update_adam(&adam_conv_biases, conv_biases, grad_biases, NUM_FILTERS, learning_rate);
        }

        printf("Epoch %d, Loss: %f, Accuracy: %f\n", epoch, epoch_loss / num_samples, epoch_accuracy / num_samples);
    }

    free_adam(&adam_fc_weights);
    free_adam(&adam_fc_biases);
    for (int i = 0; i < NUM_FILTERS; i++) {
        free_adam(&adam_filters[i]);
    }
    free_adam(&adam_conv_biases);
}

void load_mnist_data(float train_images[][INPUT_SIZE][INPUT_SIZE], float train_labels[][FC_OUTPUT_SIZE], float test_images[][INPUT_SIZE][INPUT_SIZE], float test_labels[][FC_OUTPUT_SIZE]) {
    read_mnist_images("train-images-idx3-ubyte", train_images, NUM_TRAIN);
    read_mnist_labels("train-labels-idx1-ubyte", train_labels, NUM_TRAIN);
    read_mnist_images("t10k-images-idx3-ubyte", test_images, NUM_TEST);
    read_mnist_labels("t10k-labels-idx1-ubyte", test_labels, NUM_TEST);
}

int main() {
    float (*train_images)[INPUT_SIZE][INPUT_SIZE] = (float (*)[INPUT_SIZE][INPUT_SIZE]) malloc(NUM_TRAIN * sizeof(*train_images));
    float (*train_labels)[FC_OUTPUT_SIZE] = (float (*)[FC_OUTPUT_SIZE]) malloc(NUM_TRAIN * sizeof(*train_labels));
    float (*test_images)[INPUT_SIZE][INPUT_SIZE] = (float (*)[INPUT_SIZE][INPUT_SIZE]) malloc(NUM_TEST * sizeof(*test_images));
    float (*test_labels)[FC_OUTPUT_SIZE] = (float (*)[FC_OUTPUT_SIZE]) malloc(NUM_TEST * sizeof(*test_labels));

    if (!train_images || !train_labels || !test_images || !test_labels) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    load_mnist_data(train_images, train_labels, test_images, test_labels);

    int epochs = 10;
    float learning_rate = 0.0001;

    train(train_images, train_labels, NUM_TRAIN, epochs, learning_rate);

    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);

    return 0;
}
