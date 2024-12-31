#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Struture to represent a dynamic vector for integers
typedef struct {
  int *data;       // Pointer to the dynamically allocated array
  size_t size;     // Current number of elements in the vector
  size_t capacity; // Total allocated capacity of the vector
} IntVec;

// Initial size for the dynamic vector
#define INIT_SIZE 256

/**
 * Create a new dynamic integer vector with an initial size.
 *
 * @return IntVec - A new integer vector with initialized fields.
 */
inline IntVec new_intvec() {
  IntVec vec = {
      .data = malloc(INIT_SIZE * sizeof(int)), // Allocate memory for vector
      .size = 0,                               // Initialize size to 0
      .capacity = INIT_SIZE};                  // Set initial capacity
  return vec;
}

/**
 * Append an integer to the dynamic vector, resizing if necessary.
 *
 * @param vec Pointer to the IntVec structure to which the value is appended.
 * @param i   Pointer to the integer to append.
 */
inline void vec_append(IntVec *vec, int *i) {
  // Check if resizing is needed
  if (vec->size >= vec->capacity) {
    vec->capacity *= 2; // Double the capacity
    vec->data = realloc(vec->data, vec->capacity * sizeof(*vec->data));
  }
  vec->data[vec->size++] = *i; // Add the new element and increase size
}

// Structure to hold weights for a linear model
typedef struct {
  double w; // Weight bias for feature
  double b; // Bias term
} Weights;

/**
 * Compute the gradient of the cost function for linear regression.
 *
 * @param x  Pointer to the input vector of independent variables.
 * @param y  Pointer to the output vector of targets.
 * @param ws Pointer to the current weights (w, b) of the model.
 * @return Weights - The gradients for the weight and bias (dj_dw, dj_db).
 */
Weights gradient(const IntVec *x, const IntVec *y, const Weights *ws) {
  double dj_dw = 0, dj_db = 0; // Initalize gradients for weight and bias

  // Compute the gradients for each data point
  for (size_t i = 0; i < x->size; i++) {
    double f_wb = ws->w * x->data[i] + ws->b; // Model prediction: w*x + b
    double dj_dw_i =
        (f_wb - y->data[i]) * x->data[i]; // Partial dervation w.r.t. w
    double dj_db_i = f_wb - y->data[i];   // Partial dervation w.r.t. b
    dj_dw += dj_dw_i;                     // Accumlate the gradient for w
    dj_db += dj_db_i;                     // Accumlate the gradient for b
  }

  // Average the gradients for each data point
  dj_dw = dj_dw / x->size;
  dj_db = dj_db / x->size;

  // Return the computed gradients as a Weights structure
  Weights features = {.w = dj_dw, .b = dj_db};
  return features;
}

// Main function
int main(int argc, char **argv) {

  // Check if correct number of arguments are provided
  if (argc < 2 || argc > 3) {
    // Error message explaining usage and input requirements
    fprintf(
        stderr,
        "Error: Invalid number of arguments provided.\n\nUsage: %s "
        "<input-target pairs file> or %s <input-target pairs file> <initial "
        "setting file>\n<input-target pairs file> (input-target.txt) example: "
        "\n1 2\n2 3\n3 4\n123 432\n10 1\n-10 37\n\n<initial settings file> "
        "(settings.txt) example:\nw 0.0\nb 0.0\nalpha 0.00001\niterations "
        "100000\noutput stdout\nlog-every 100\n\nSettings file explanation:\nw "
        "= initial weight, b = initial bias, alpha = learning "
        "rate,\niterations = number of iterations to train (inclusive) "
        "starting from 0\n(e.g. 1000 would be 0..1000 or 1001 total, 10000 "
        "would be 0..10000 or 10001 total),\nlog-every = number of iterations "
        "to pass between before logging (e.g. log-every 100 would log 0 100 "
        "200 ...),\noutput = file where the output will be written (left "
        "unspecified uses stdout)\n\nIt is fine to not provide a initial "
        "settings file, if one is not provided,\nthe settings listed in the "
        "example will be used.\nFurthermore, you don't have to specify all the "
        "settings in your <initial settings file>\nand the ordering of your "
        "settings does not matter.\n\nAnother <initial settings file> "
        "(settings.txt) example:\nlog-every 1000\nw 100\n\nIs also a valid "
        "settings file.\n\nFiles should be txts with the format value <space> "
        "value and should be in the same directory as the executable.\n",
        argv[0], argv[0]);
    return 1;
  }

  // Open the input-target pairs file
  FILE *input_target_pair_file = fopen(argv[1], "r");
  if (input_target_pair_file == NULL) {
    // Error message if file can not be opened
    perror("Error opening target-value file");
    return 1;
  }

  // Temporary variables to store pairs read from the file
  int x_temp, y_temp;

  // Initialize dynamic vectors for x (inputs) and y (targets)
  IntVec x = new_intvec();
  IntVec y = new_intvec();

  // Read pairs of integers from the file and store them in vectors
  while (fscanf(input_target_pair_file, "%d %d", &x_temp, &y_temp) == 2) {
    vec_append(&x, &x_temp); // Append the first integer to the x vector
    vec_append(&y, &y_temp); // Append the second integer to the y vector
  }

  // Close the input-target pairs file
  fclose(input_target_pair_file);

  // Default initial settings
  double w = 0.0, b = 0.0, alpha = 0.00001;
  int iterations = 100000, every = 100;
  char output[101] = "";

  // Check if optional initial settings file is provided
  if (argc == 3) {
    FILE *settings_file = fopen(argv[2], "r");
    if (settings_file == NULL) {
      // Error message if the settings file cannot be opened
      perror("Error opening settings file");

      // Free allocated memory before exiting
      free(x.data);
      free(y.data);
      return 1;
    }

    // Temporary variables to store keys and values from the settings file
    char key[11]; // Buffer to store the setting key (e.g., "w", "b", "alpha")
    char value[101]; // Variable to store the corresponding value

    // Read key-value pairs from the settings file
    while (fscanf(settings_file, "%10s %100s", key, value) == 2) {
      // Match the key and update the corresponding variable
      if (strcmp(key, "w") == 0)
        w = atof(value);
      else if (strcmp(key, "b") == 0)
        b = atof(value);
      else if (strcmp(key, "alpha") == 0)
        alpha = atof(value);
      else if (strcmp(key, "iterations") == 0)
        iterations = atof(value);
      else if (strcmp(key, "log-every") == 0)
        every = atof(value);
      else if (strcmp(key, "output") == 0) {
        strncpy(output, value, sizeof(output) - 1);
        output[sizeof(output) - 1] = '\0';
      } else
        fprintf(stderr, "Unknown key: %s\n", key);
    }

    // Close the settings file
    fclose(settings_file);
  }

  // Create output file pointer
  FILE *output_file = NULL;
  // If a specified output file was provided, open it
  if (strcmp(output, "") != 0) {
    output_file = fopen(output, "w");
    // Error message if the output file could not be opened
    if (output_file == NULL) {
      perror("Error Opening File");

      // Free allocated memeory before exiting
      free(x.data);
      free(y.data);
      return 1;
    }
  }

  // Initialize weights with the specified or default values
  Weights weights = {.w = w, .b = b};

  // Training loop to update weights over the specified number of iterations
  for (int i = 0; i <= iterations; i++) {
    // Compute the gradient for the current weights
    Weights ws = gradient(&x, &y, &weights);

    // Update weights using the learning rate and gradient
    weights.w = weights.w - alpha * ws.w;
    weights.b = weights.b - alpha * ws.b;

    // Print the weights every specified number of iterations for progress
    // tracking
    if (i % every == 0) {
      // Write to the output file if there is one open
      if (output_file != NULL)
        fprintf(output_file, "iteration: %d, w: %lf, b: %lf\n", i, weights.w,
                weights.b);
      // Otherwise write to stdout
      else
        fprintf(stdout, "iteration: %d, w: %lf, b: %lf\n", i, weights.w,
                weights.b);
    }
  }

  // Close the output file
  fclose(output_file);

  // Free dynamically allocated memory for vectors
  free(x.data);
  free(y.data);

  // Exit successfully
  return 0;
}
