#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <immintrin.h> // For SIMD intrinsics


void naive_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size);
void simd_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size);

/**
 * @brief 		Generates random numbers between values fMin and fMax.
 * @param 		fMin 	lower range
 * @param 		fMax 	upper range
 * @return 		random floating point number
 */
double fRand(double fMin, double fMax)
{

    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

/**
 * @brief 		Initialize a matrix of given dimension with random values.
 * @param 		matrix 		pointer to the matrix
 * @param 		rows 		number of rows in the matrix
 * @param 		cols 		number of columns in the matrix
 */
void initialize_kernel(double *matrix, int rows, int cols)
{

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i * cols + j] = ceil(fRand(0.0001, 1.0000)); // random values between 0 and 1
        }
    }
}

/**
 * @brief 		Initialize a matrix of given dimension with random values.
 * @param 		matrix 		pointer to the matrix
 * @param 		rows 		number of rows in the matrix
 * @param 		cols 		number of columns in the matrix
 */
void initialize_matrix(double *matrix, int rows, int cols)
{

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i * cols + j] = fRand(0.0001, 1.0000); // random values between 0 and 1
        }
    }
}

/**
 * @brief 		Initialize result matrix of given dimension with 0.
 * @param 		matrix 		pointer to the matrix
 * @param 		rows 		number of rows in the matrix
 * @param 		cols 		number of columns in the matrix
 */
void initialize_result_matrix(double *matrix, int rows, int cols)
{

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i * cols + j] = 0.0;
        }
    }
}

void verify_correctness(double *C, double *D, int dim)
{
    double epsilon = 1e-9;
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            if (fabs(C[i * dim + j] - D[i * dim + j]) > epsilon)
            {
                printf("%f & %f at location (%d %d)\n", C[i * dim + j], D[i * dim + j], i, j);
                printf("Matrix convolution is incorrect!\n");
                return;
            }
        }
    }
    printf("Matrix convolution is correct!\n");
    return;
}

double measure_execution_time(void (*func)(double *, double *, double *, int, int, int), double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size);

int main(int argc, char **argv)
{
    if (argc <= 2)
    {
        printf("Usage: matrix-dimension kernel-size\n\n");
        return 0;
    }

    int dim = atoi(argv[1]);
    int kernel_size = atoi(argv[2]);
    int output_dim = dim - kernel_size + 1;

    // Allocate memory for the input and output images
    double *input_image = (double *)malloc(dim * dim * sizeof(double));
    double *output_image = (double *)malloc(output_dim * output_dim * sizeof(double));
    double *kernel = (double *)malloc(kernel_size * kernel_size * sizeof(double));
    double *optimized_op = (double *)malloc(output_dim * output_dim * sizeof(double));

    // Initialize the input image and kernel
    initialize_matrix(input_image, dim, dim);

    // Initialize the kernel
    initialize_kernel(kernel, kernel_size, kernel_size);

    // Initialize the output image
    initialize_result_matrix(output_image, output_dim, output_dim);

    // Measure execution time and perform naive convolution
    double naive_time = measure_execution_time(naive_convolution, input_image, output_image, kernel, dim, output_dim, kernel_size);

    // Print the execution times and speedups
    printf("Naive Convolution Time: %f seconds\n", naive_time);


    // Measure execution time and perform SIMD convolution
    #ifdef OPTIMIZE_SIMD

    initialize_result_matrix(optimized_op, output_dim, output_dim);

    double simd_time = measure_execution_time(simd_convolution, input_image, optimized_op, kernel, dim, output_dim, kernel_size);
    double simd_speedup = naive_time / simd_time;


    printf("SIMD Convolution Time: %f seconds, Speedup: %fx\n", simd_time, simd_speedup);
    verify_correctness(output_image, optimized_op, output_dim);

    #endif



    // Free allocated memory
    free(input_image);
    free(output_image);
    free(optimized_op);

    return 0;
}

// Naive convolution implementation
void naive_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size)
{
    for (int i = 0; i < output_dim; i++)
    {
        for (int j = 0; j < output_dim; j++)
        {
            // printf("i: %d, j: %d\n", i, j);
            double sum = 0.0;
            for (int ki = 0; ki < kernel_size; ki++)
            {
                for (int kj = 0; kj < kernel_size; kj++)
                {
                    int x = i + ki;
                    int y = j + kj;
                    sum += input_image[x * dim + y] * kernel[ki * kernel_size + kj];
                }
            }
            output_image[i * output_dim + j] = sum;
        }
    }
}

// SIMD convolution implementation
void simd_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size)
{
    // students need to implement this function
}

// Function to measure execution time of a convolution function
double measure_execution_time(void (*func)(double *, double *, double *, int, int, int), double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size)
{
    clock_t start, end;
    start = clock();
    func(input_image, output_image, kernel, dim, output_dim, kernel_size);
    end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}
