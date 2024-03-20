#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define MNIST_IMAGE_SIZE 28 * 28
#define MNIST_NUM_IMAGES 60000
#define MNIST_NUM_LABELS 10
#define MNIST_NUM_IMAGES_TEST 9000

double* relu(double* x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0) {
            x[i] = 0;
        }
    }
    return x;
}

double* softmax(double* x, int size) {
    double max = x[0];
    double x_sum = 0;
    for (int i = 1; i < size; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max);
    }
    // synchronize
    for (int i = 0; i < size; i++) {
        x_sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] = x[i] / x_sum;
    }
    return x;
}

double categorical_cross_entropy_loss(double* x, double* target, int size) {
    double res = 0;
    for (int i = 0; i < size; i++) {
        res += target[i] * log(x[i] + 0.000000000000001);
    }
    return res;
}

void layer_builder(int size, int size_input, double** l_p, double** l_b) {
    *l_p = (double*)malloc(size * size_input * sizeof(double));
    *l_b = (double*)malloc(size * sizeof(double));

    for (int i = 0; i < size * size_input; i++) {
        (*l_p)[i] = rand() / (RAND_MAX + 1.0);
    }
    for (int i = 0; i < size; i++) {
        (*l_b)[i] = rand() / (RAND_MAX + 1.0);
    }

    printf("Number of params for layer: %d\n", size * size_input);
}

double* layer_inf(double* l_p, double* l_b, int num_outputs, double* neurons, int num_inputs) {
    double* output = (double*)malloc(num_outputs * sizeof(double));

    for (int j = 0; j < num_outputs; j++) {
        output[j] = l_b[j];
        for (int i = 0; i < num_inputs; i++) {
            output[j] += neurons[i] * l_p[j * num_inputs + i];
        }
    }

    return output;
}

void read_mnist_images(const char* image_file_path, uint8_t** images, int num_images) {
    FILE* file = fopen(image_file_path, "rb");
    if (file == NULL) {
        printf("Error opening file: %s\n", image_file_path);
        return;
    }

    uint32_t magic_number, num_images_file;
    fread(&magic_number, sizeof(uint32_t), 1, file);
    fread(&num_images_file, sizeof(uint32_t), 1, file);

    uint32_t rows, cols;
    fread(&rows, sizeof(uint32_t), 1, file);
    fread(&cols, sizeof(uint32_t), 1, file);

    for (int i = 0; i < num_images; i++) {
        fread(images[i], sizeof(uint8_t), MNIST_IMAGE_SIZE, file);
    }

    fclose(file);
}

void read_mnist_labels(const char* label_file_path, uint8_t* labels, uint8_t** one_hot_labels, int num_labels) {
    FILE* file = fopen(label_file_path, "rb");
    if (file == NULL) {
        printf("Error opening file: %s\n", label_file_path);
        return;
    }

    uint32_t magic_number, num_labels_file;
    fread(&magic_number, sizeof(uint32_t), 1, file);
    fread(&num_labels_file, sizeof(uint32_t), 1, file);

    for (int i = 0; i < num_labels; i++) {
        uint8_t label;
        fread(&label, sizeof(uint8_t), 1, file);
        labels[i] = label;
        for (int j = 0; j < MNIST_NUM_LABELS; j++) {
            one_hot_labels[i][j] = (j == label) ? 1 : 0;
        }
    }

    fclose(file);
}

void custom_MLP_2l_build(int size_input, int size_l2, int size_output, double** l1p, double** l1b, double** l2p, double** l2b) {
    printf("Build layer 1\n");
    layer_builder(size_l2, size_input, l1p, l1b);
    printf("Build layer 2\n");
    layer_builder(size_output, size_l2, l2p, l2b);
}

double* custom_MLP_2l_inf(double* input_data, int size_input, int size_l2, int size_output, double* l1p, double* l1b, double* l2p, double* l2b) {
    double* n1 = (double*)malloc(size_l2 * sizeof(double));
    double* n2 = (double*)malloc(size_output * sizeof(double));

    n1 = layer_inf(l1p, l1b, size_l2, input_data, size_input);
    n1 = relu(n1, size_l2);
    n2 = layer_inf(l2p, l2b, size_output, n1, size_l2);
    n2 = softmax(n2, size_output);

    free(n1);
    return n2;
}

double evaluate_accuracy(double* l1p, double* l1b, double* l2p, double* l2b, int size_input, int size_l2, int size_output) {
    uint8_t** test_images = (uint8_t**)malloc(MNIST_NUM_IMAGES_TEST* sizeof(uint8_t*));
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        test_images[i] = (uint8_t*)malloc(MNIST_IMAGE_SIZE * sizeof(uint8_t));
    }
    uint8_t** one_hot_labels = (uint8_t**)malloc(MNIST_NUM_IMAGES_TEST * sizeof(uint8_t*));
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        one_hot_labels[i] = (uint8_t*)malloc(MNIST_NUM_LABELS * sizeof(uint8_t));
    }
    uint8_t* test_labels = (uint8_t*)malloc(MNIST_NUM_IMAGES_TEST * sizeof(uint8_t));

    read_mnist_images("train-images.idx3-ubyte", test_images, MNIST_NUM_IMAGES_TEST);
    read_mnist_labels("train-labels.idx1-ubyte", test_labels, one_hot_labels, MNIST_NUM_IMAGES_TEST);
    double correct_predictions = 0.0;
    double* input_data = (double*)malloc(MNIST_IMAGE_SIZE * sizeof(double));
    double* output;

    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            input_data[j] = (double)(test_images[i][j]) / 255.0;
        }
        output = custom_MLP_2l_inf(input_data, MNIST_IMAGE_SIZE, size_l2, size_output, l1p, l1b, l2p, l2b);
        int predicted_label = 0;
        double max_output = output[0];
        for (int j = 1; j < size_output; j++) {
            if (output[j] > max_output) {
                max_output = output[j];
                predicted_label = j;
            }
        }

        int true_label = 0;
        for (int j = 0; j < MNIST_NUM_LABELS; j++) {
            if (one_hot_labels[i][j] == 1) {
                true_label = j;
                break;
            }
        }

        if (predicted_label == true_label) {
            correct_predictions++;
        }
    }
    free(input_data);
    free(test_images);
    free(test_labels);
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        free(one_hot_labels[i]);
    }
    free(one_hot_labels);

    return correct_predictions / MNIST_NUM_IMAGES_TEST;
}


int main() {
        
    srand(66);  
    uint8_t** images = (uint8_t**)malloc(MNIST_NUM_IMAGES_TEST * sizeof(uint8_t*));
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        images[i] = (uint8_t*)malloc(MNIST_IMAGE_SIZE * sizeof(uint8_t));
    }
    uint8_t** one_hot_labels = (uint8_t**)malloc(MNIST_NUM_IMAGES_TEST * sizeof(uint8_t*));
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        one_hot_labels[i] = (uint8_t*)malloc(MNIST_NUM_LABELS * sizeof(uint8_t));
    }
    uint8_t* labels = (uint8_t*)malloc(MNIST_NUM_IMAGES_TEST * sizeof(uint8_t));

    printf("Loading train\n");
    read_mnist_images("train-images.idx3-ubyte", images, MNIST_NUM_IMAGES_TEST);
    printf("Loading train labels\n");
    read_mnist_labels("train-labels.idx1-ubyte", labels, one_hot_labels, MNIST_NUM_IMAGES_TEST);
    printf("Build model\n");

    // Use the loaded images and one-hot encoded labels for training or other purposes
    double *l1p, *l1b, *l2p, *l2b;
    custom_MLP_2l_build(MNIST_IMAGE_SIZE, 128, MNIST_NUM_LABELS, &l1p, &l1b, &l2p, &l2b);

    printf("Evaluate model\n");
    double accuracy = evaluate_accuracy(l1p, l1b, l2p, l2b, MNIST_IMAGE_SIZE, 128, MNIST_NUM_LABELS);
    printf("Accuracy: %f\n", accuracy);

    // Free the allocated memory
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        free(images[i]);
        free(one_hot_labels[i]);
    }
    free(images);
    free(one_hot_labels);
    free(labels);
    free(l1p);
    free(l1b);
    free(l2p);
    free(l2b);

    return 0;
}
