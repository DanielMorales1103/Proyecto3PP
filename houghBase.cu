/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
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
//*****************************************************************
// The CPU function returns a pointer to the accummulator
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

//*****************************************************************
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
//__constant__ float d_Cos[degreeBins];
//__constant__ float d_Sin[degreeBins];

//*****************************************************************
//TODO Kernel memoria compartida
// __global__ void GPU_HoughTranShared(...)
// {
//   //TODO
// }
//TODO Kernel memoria Constante
// __global__ void GPU_HoughTranConst(...)
// {
//   //TODO
// }

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
  //TODO calcular: int gloID = ?
  int gloID = blockIdx.x * blockDim.x + threadIdx.x; //TODO
  if (gloID >= w * h) return;      // in case of extra threads in block

  int xCent = w / 2;
  int yCent = h / 2;

  //TODO explicar bien bien esta parte. Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  //TODO eventualmente usar memoria compartida para el acumulador

  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          // Usar valores precomputados de coseno y seno
          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          //debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
          atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
        }
    }

  //TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
  //utilizar operaciones atomicas para seguridad
  //faltara sincronizar los hilos del bloque en algunos lados

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
        // Marcar el píxel si está dentro de los límites
        if (x1 >= 0 && x1 < w && y1 >= 0 && y1 < h)
        {
            image[y1 * w + x1] = 255; // Valor blanco para las líneas
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

__global__ void drawDetectedLines(int *acc, int rBins, int degreeBins, int w, int h, float rMax, float rScale, float *d_Cos, float *d_Sin, unsigned char *image, int threshold, float radInc)
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


//*****************************************************************
int main (int argc, char **argv)
{
  int i;

  PGMImage inImg (argv[1]);

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  float* d_Cos;
  float* d_Sin;

  cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // pre-compute values to be stored
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // TODO eventualmente volver memoria global
  cudaMemcpy(d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels; // h_in contiene los pixeles de la imagen
  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  //1 thread por pixel
  int kernelBlockNum  = max(1, (int)ceil((float)(w * h) / 256));
  printf("blockNum: %d, w: %d, h: %d\n", kernelBlockNum , w, h);

  // CUDA Events para medir el tiempo
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Registrar inicio del evento
  cudaEventRecord(start, 0);
  //lanzar el kernel
  GPU_HoughTran <<< kernelBlockNum , 256 >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
  //Registra fin del evento
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
  printf("Tiempo de ejecución del kernel: %.2f ms\n", elapsedTime);

  // Destruir eventos CUDA
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // get results from device
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);


  // Calcular threshold basado en el promedio y desviación estándar
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
  int threshold = mean + 2 * stddev;

  printf("Threshold calculado: %d\n", threshold);

  // Crear buffer para la imagen en la GPU
  unsigned char *d_image, *h_image;
  h_image = (unsigned char *)calloc(w * h, sizeof(unsigned char)); // Inicializa en negro
  cudaMalloc((void **)&d_image, sizeof(unsigned char) * w * h);
  cudaMemset(d_image, 0, sizeof(unsigned char) * w * h); // Inicializar en negro

  // Lanzar kernel para dibujar las líneas
  int totalThreads = rBins * degreeBins;
  int drawBlockNum = (totalThreads + 255) / 256;
  drawDetectedLines<<<drawBlockNum, 256>>>(d_hough, rBins, degreeBins, w, h, rMax, rScale, d_Cos, d_Sin, d_image, threshold, radInc);
  cudaDeviceSynchronize();

  // Copiar la imagen resultante de la GPU al host
  cudaMemcpy(h_image, d_image, sizeof(unsigned char) * w * h, cudaMemcpyDeviceToHost);

  // Guardar la imagen como archivo PNG
  stbi_write_png("output.png", w, h, 1, h_image, w);
  printf("Imagen con líneas detectadas guardada como output.png\n");

  // Liberar memoria de la imagen
  cudaFree(d_image);
  free(h_image);


  // compare CPU and GPU results
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
      printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }
  printf("Done!\n");

  // Liberar memoria de la GPU
  cudaFree(d_Cos);
  cudaFree(d_Sin);
  cudaFree(d_in);
  cudaFree(d_hough);
  // Liberar memoria del host
  free(pcCos);
  free(pcSin);
  free(h_hough);

  printf("Memoria liberada correctamente.\n");

  return 0;
}
