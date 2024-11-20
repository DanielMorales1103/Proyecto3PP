#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

__constant__ float d_CosConst[degreeBins];
__constant__ float d_SinConst[degreeBins];


void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) //por cada pixel
    for (int j = 0; j < h; j++) //...
      {
        int idx = j * w + i;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            float theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                theta += radInc;
              }
          }
      }
}

__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    int locID = threadIdx.x;

    if (gloID >= w * h) return;

    __shared__ int localAcc[degreeBins * rBins];

    // Inicializar acumulador local a 0
    for (int i = locID; i < degreeBins * rBins; i += blockDim.x)
    {
        localAcc[i] = 0;
    }
    __syncthreads();

    int xCent = w / 2;
    int yCent = h / 2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0){
        for (int tIdx = 0; tIdx < degreeBins; tIdx++){
            float r = xCoord * d_CosConst[tIdx] + yCoord * d_SinConst[tIdx];
            int rIdx = (r + rMax) / rScale;
            if (rIdx >= 0 && rIdx < rBins){
                atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
            }
        }
    }
    __syncthreads(); // Barrera: Esperar a que todos completen la actualización de localAcc
    
    // Transferir valores de localAcc a acc global
    for (int i = locID; i < degreeBins * rBins; i += blockDim.x){
        atomicAdd(&acc[i], localAcc[i]);
    }
}


__device__ void drawLine(int x1, int y1, int x2, int y2, unsigned char *image, int w, int h)
{
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;

    while (true)
    {
        if (x1 >= 0 && x1 < w && y1 >= 0 && y1 < h)
        {
            int idx = (y1 * w + x1) * 3;
            image[idx + 0] = 255; // Rojo
            image[idx + 1] = 0;   // Verde
            image[idx + 2] = 0;   // Azul
        }

        if (x1 == x2 && y1 == y2)
            break;

        int e2 = 2 * err;
        if (e2 > -dy)
        {
            err -= dy;
            x1 += sx;
        }
        if (e2 < dx)
        {
            err += dx;
            y1 += sy;
        }
    }
}

__global__ void drawDetectedLines(int *acc, int rBins, int degreeBins, int w, int h, float rMax, float rScale, unsigned char *image, int threshold, float radInc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rBins * degreeBins) return;

    int rIdx = idx / degreeBins;
    int tIdx = idx % degreeBins;

    // Verificar si el peso supera el threshold
    int weight = acc[rIdx * degreeBins + tIdx];
    if (weight > threshold)
    {
        // Calcular r y θ usando radInc
        float r = rIdx * rScale - rMax;
        float theta = tIdx * radInc;

        // Convertir a coordenadas de línea
        float a = cos(theta);
        float b = sin(theta);
        float x0 = a * r;
        float y0 = b * r;
        int x1 = round(x0 + 1000 * (-b));
        int y1 = round(y0 + 1000 * (a));
        int x2 = round(x0 - 1000 * (-b));
        int y2 = round(y0 - 1000 * (a));

        // Ajustar al sistema centrado
        x1 += w / 2;
        x2 += w / 2;
        y1 = h / 2 - y1;
        y2 = h / 2 - y2;

        // Dibujar la línea en el buffer de imagen
        drawLine(x1, y1, x2, y2, image, w, h);
    }
}


int main(int argc, char **argv)
{
    int i;

    PGMImage inImg(argv[1]);

    int *cpuht;
    int w = inImg.x_dim;
    int h = inImg.y_dim;

    // CPU Calculation
    CPU_HoughTran(inImg.pixels, w, h, &cpuht);

    // Pre-compute values for sin and cos
    float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (i = 0; i < degreeBins; i++)
    {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    // Transfer precomputed values to constant memory
    cudaMemcpyToSymbol(d_CosConst, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_SinConst, pcSin, sizeof(float) * degreeBins);

    // Setup and copy data from host to device
    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = inImg.pixels;
    h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    // Execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
    int kernelBlockNum = max(1, (int)ceil((float)(w * h) / 256));

    // CUDA Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel and measure execution time
    cudaEventRecord(start, 0);
    GPU_HoughTran<<<kernelBlockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);
    cudaEventRecord(stop, 0);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Tiempo de ejecucion del kernel: %.2f ms\n", elapsedTime);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Get results from device
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // Calculate threshold based on mean and standard deviation
    int sum = 0, count = 0;
    for (i = 0; i < rBins * degreeBins; i++)
    {
        sum += h_hough[i];
        if (h_hough[i] > 0)
            count++;
    }
    float mean = sum / (float)count;
    float stddev = 0;
    for (i = 0; i < rBins * degreeBins; i++)
    {
        if (h_hough[i] > 0)
            stddev += pow(h_hough[i] - mean, 2);
    }
    stddev = sqrt(stddev / count);
    int threshold = mean + 2.2 * stddev;

    printf("Threshold calculado: %d\n", threshold);

    // Create buffer for image in GPU
    unsigned char *d_image, *h_image;
    h_image = (unsigned char *)calloc(w * h * 3, sizeof(unsigned char)); // Initialize to black
    cudaMalloc((void **)&d_image, sizeof(unsigned char) * w * h * 3);

    // Convert the original image to RGB
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            int idxGray = i * w + j;
            int idxRGB = idxGray * 3;
            unsigned char pixel = h_in[idxGray];
            h_image[idxRGB + 0] = pixel; // Red
            h_image[idxRGB + 1] = pixel; // Green
            h_image[idxRGB + 2] = pixel; // Blue
        }
    }

    // Copy initial image to GPU
    cudaMemcpy(d_image, h_image, sizeof(unsigned char) * w * h * 3, cudaMemcpyHostToDevice);

    int totalThreads = rBins * degreeBins;
    int drawBlockNum = (totalThreads + 255) / 256;

    // Launch kernel to draw detected lines
    drawDetectedLines<<<drawBlockNum, 256>>>(d_hough, rBins, degreeBins, w, h, rMax, rScale, d_image, threshold, radInc);
    cudaDeviceSynchronize();

    // Copy resulting image from GPU to host
    cudaMemcpy(h_image, d_image, sizeof(unsigned char) * w * h * 3, cudaMemcpyDeviceToHost);

    // Save the image as PNG
    stbi_write_png("outputcomp.png", w, h, 3, h_image, w * 3);
    printf("Imagen con lineas detectadas guardada como outputcomp.png\n");

    // Free image memory
    cudaFree(d_image);
    free(h_image);

    printf("Done!\n");

    // Free GPU memory
    cudaFree(d_in);
    cudaFree(d_hough);

    // Free host memory
    free(pcCos);
    free(pcSin);
    free(h_hough);

    printf("Memoria liberada correctamente.\n");

    return 0;
}

