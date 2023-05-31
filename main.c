#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define TRAIN_DATA_SET_SIZE 60000
#define TEST_DATA_SET_SIZE 10000
#define IMAGE_SIZE 784
#define HIDDEN_LAYER_SIZE 10
#define OUTPUT_LAYER_SIZE 10
#define MINIBATCH_SIZE 100
#define TOTAL_EPOCHS 1000
#define LEARNING_RATE 0.01

void ReadData(double** dataset, int* trainLabel, char* fileName)
{
    FILE* fp = fopen(fileName, "r");
    if (!fp) printf("Can't open file\n");
    else
    {
        char buffer[10000];
 
        int row = 0;
        int column;

        fgets(buffer, 10000, fp);
 
        while (fgets(buffer, 10000, fp))
        {
            column = 0;

            char* value = strtok(buffer, ",");

            trainLabel[row] = atoi(value);
 
            while (value)
            {
                value = strtok(NULL, ",");
                
                if(value != NULL)
                    dataset[row][column] = (double) atoi(value) / 255.0;

                column++;
            }

            row++;
        }
    }

    fclose(fp);

    return;
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double sigmoidDerivative(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

void ForwardPropagation(double* image, double* layer0, double* layer1, double** w0, double* b0, double** w1, double* b1)
{

    for (size_t i = 0; i < HIDDEN_LAYER_SIZE; i++)
        layer0[i] = 0;
    for (size_t i = 0; i < HIDDEN_LAYER_SIZE; i++)
    {
        for (size_t j = 0; j < IMAGE_SIZE; j++)
            layer0[i] += image[j] * w0[i][j];
        layer0[i] += b0[i];
    }
    
    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
        layer1[i] = 0;
    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
    {
        for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
            layer1[i] += sigmoid(layer0[j]) * w1[i][j];
        layer1[i] += b1[i];
    }

    return;
    
}

void BackPropagation(double* image, int trainLabel, double* layer0, double* layer1, double** w0, double* b0, double** w1, double* b1)
{
    double delta1[OUTPUT_LAYER_SIZE];
    double delta0[HIDDEN_LAYER_SIZE];

    int trainLabelArray[OUTPUT_LAYER_SIZE];

    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
    {
        if(trainLabel == i)
            trainLabelArray[i] = 1;
        else
            trainLabelArray[i] = 0;
    }
    

    //Calculando os deltas

    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
        delta1[i] = LEARNING_RATE * (2 * (sigmoid(layer1[i]) - trainLabelArray[i]) / OUTPUT_LAYER_SIZE) * (sigmoid(layer1[i]) * (1 - sigmoid(layer1[i])));

    for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
        delta0[j] = 0;

    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
        for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
            delta0[j] += delta1[i] * w1[i][j];
    
    for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
        delta0[j] *= (sigmoid(layer0[j]) * (1 - sigmoid(layer0[j])));

    //Modificando os valores

    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
        b1[i] -= delta1[i];

    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
        for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
            w1[i][j] -= delta1[i] * (sigmoid(layer0[j]));

    for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
        b0[j] -= delta0[j];

    for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
        for (size_t k = 0; k < IMAGE_SIZE; k++)
            w0[j][k] -= delta0[j] * image[k];
    
    return;
}

int main()
{
    printf("INITIALIZING...\n");

    srand(time(NULL));

    printf("ALLOCATING MEMORY SPACE...\n");

    double** trainDataset = (double**) malloc(sizeof(double*) * TRAIN_DATA_SET_SIZE);
    for (size_t i = 0; i < TRAIN_DATA_SET_SIZE; i++)
        trainDataset[i] = (double*) malloc(sizeof(double) * IMAGE_SIZE);
    
    int* trainLabel = (int*) malloc(sizeof(int*)* TRAIN_DATA_SET_SIZE);

    double** testDataset = (double**) malloc(sizeof(double*) * TEST_DATA_SET_SIZE);
    for (size_t i = 0; i < TEST_DATA_SET_SIZE; i++)
        testDataset[i] = (double*) malloc(sizeof(double) * IMAGE_SIZE);
    
    int* testLabel = (int*) malloc(sizeof(int*)* TEST_DATA_SET_SIZE);

    double* layer0 = (double*) malloc(sizeof(double) * HIDDEN_LAYER_SIZE);
    double* layer1 = (double*) malloc(sizeof(double) * OUTPUT_LAYER_SIZE);

    double* b0 = (double*) malloc(sizeof(double) * HIDDEN_LAYER_SIZE);
    double* b1 = (double*) malloc(sizeof(double) * OUTPUT_LAYER_SIZE);
    
    double** w0 = (double**) malloc(sizeof(double*) * HIDDEN_LAYER_SIZE);
    for (size_t i = 0; i < HIDDEN_LAYER_SIZE; i++)
        w0[i] = (double*) malloc(sizeof(double) * IMAGE_SIZE);

    double** w1 = (double**) malloc(sizeof(double*) * OUTPUT_LAYER_SIZE);
    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
        w1[i] = (double*) malloc(sizeof(double) * HIDDEN_LAYER_SIZE);   

    printf("GENERATING INITIAL RANDOM VALUES...\n");

    for (size_t i = 0; i < HIDDEN_LAYER_SIZE; i++)
        for (size_t j = 0; j < IMAGE_SIZE; j++)
            w0[i][j] = ((double) rand() / (double) RAND_MAX) - 0.5;

    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
        for (size_t j = 0; j < HIDDEN_LAYER_SIZE; j++)
            w1[i][j] = ((double) rand() / (double) RAND_MAX) - 0.5;

    for (size_t i = 0; i < HIDDEN_LAYER_SIZE; i++)
        b0[i] = ((double) rand() / (double) RAND_MAX) - 0.5;

    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
        b1[i] = ((double) rand() / (double) RAND_MAX) - 0.5;

    printf("READING TRAIN DATASET...\n");

    ReadData(trainDataset, trainLabel, "./data/mnist_train.csv");
    ReadData(testDataset, testLabel, "./data/mnist_test.csv");

    printf("EXECUTING FORWARDPROPAGATION/BACKPROPAGATION...\n");

    for (size_t epoch = 0; epoch < TOTAL_EPOCHS; epoch++)
    {
        clock_t begin = clock();

        double costFunction = 0;

        for (size_t k = 0; k < TRAIN_DATA_SET_SIZE; k++)
        {
            ForwardPropagation(trainDataset[k], layer0, layer1, w0, b0, w1, b1);

            double lossFunction = 0;

            int trainLabelArray[OUTPUT_LAYER_SIZE];

            for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
            {
                if(trainLabel[k] == i)
                    trainLabelArray[i] = 1;
                else
                    trainLabelArray[i] = 0;
            }

            for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
                lossFunction += pow(sigmoid(layer1[i]) - trainLabelArray[i], 2) / OUTPUT_LAYER_SIZE;
            
            costFunction += lossFunction;
        
            BackPropagation(trainDataset[k], trainLabel[k], layer0, layer1, w0, b0, w1, b1);
        }

        int correctGuess = 0;
            
        for (size_t k = 0; k < TEST_DATA_SET_SIZE; k++)
        {
            ForwardPropagation(testDataset[k], layer0, layer1, w0, b0, w1, b1);

            int biggest = 0;

            for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
                if(layer1[biggest] < layer1[i])
                    biggest = i;
            
            if(testLabel[k] == biggest)
                correctGuess++;
        }

        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

        printf("EPOCH:%d\tCOST:%.2f\tTIME:%.1f sec\tACCURACY:%.2f%%\n", epoch, costFunction, time_spent, (double)correctGuess*100/TEST_DATA_SET_SIZE);
    }

    printf("FINALIZING...\n");

    for (size_t i = 0; i < TRAIN_DATA_SET_SIZE; i++)
        free(trainDataset[i]);
    free(trainDataset);
    free(trainLabel);

    for (size_t i = 0; i < TEST_DATA_SET_SIZE; i++)
        free(testDataset[i]);
    free(testDataset);
    free(testLabel);

    free(layer0);
    free(layer1);
    free(b0);
    free(b1);

    for (size_t i = 0; i < HIDDEN_LAYER_SIZE; i++)
        free(w0[i]);
    free(w0);

    for (size_t i = 0; i < OUTPUT_LAYER_SIZE; i++)
        free(w1[i]);
    free(w1);

    printf("SUCCESS\n");

    return 0;
}