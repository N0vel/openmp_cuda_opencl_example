#include "cuda_runtime.h"
#include <iostream>
#include "bitmap_image.hpp"
#include <ctime>
#include <stdio.h>
using namespace std;

#define BSIZE 16
#define input_image_size 2

void cudaCheckErrors(string msg)
{
	while (0)
	{
		cudaError_t __err = cudaGetLastError();
		if (__err != cudaSuccess)
		{
		fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__);
		fprintf(stderr, "*** FAILED - ABORTING\n");
		system("pause");
		exit(1);
		}
	}
}

void check_cuda_devices()
{
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
		prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
		prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
		2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
	}

}

__global__ void gauss_filter(float *input, float *output, int columns, int rows)
{
	int row = threadIdx.y + blockDim.y*blockIdx.y;
	int col = threadIdx.x + blockDim.x*blockIdx.x;
	if ((row > 0) && (row < rows - 1) && (col > 0) && (col < columns-1))
	{
		output[(row-1)  * (columns-2) + (col-1)] = 0.0;
		output[(row-1)  * (columns-2) + (col-1)] += 1./16. * input[(row-1) * columns + col - 1];
		output[(row-1)  * (columns-2) + (col-1)] += 2./16. * input[(row-1) * columns + col];
		output[(row-1)  * (columns-2) + (col-1)] += 1./16. * input[(row-1) * columns + col + 1];
		output[(row-1)  * (columns-2) + (col-1)] += 2./16. * input[row * columns + col - 1];
		output[(row-1)  * (columns-2) + (col-1)] += 4./16. * input[row * columns + col];
		output[(row-1)  * (columns-2) + (col-1)] += 2./16. * input[row * columns + col + 1];
		output[(row-1)  * (columns-2) + (col-1)] += 1./16. * input[(row+1) * columns + col - 1];
		output[(row-1)  * (columns-2) + (col-1)] += 2./16. * input[(row+1) * columns + col];
		output[(row-1)  * (columns-2) + (col-1)] += 1./16. * input[(row+1) * columns + col + 1];
	}
}

float* rgb_to_gray(bitmap_image image)
{
	int h = image.height();
	int w = image.width();
    unsigned char r, g, b;
	float *gray = new float[w*h];
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			image.get_pixel(x, y, r, g, b);
			gray[y*w+x] = (float) ((r+g+b) / 3.);
		}

	}
	return gray;
}


int main() 
{
	check_cuda_devices();
	cout << "BLOCK SIZE: "  << BSIZE << endl;
	cout << "Image index: " << input_image_size << endl;
    char images [3][50] = {"1024x768.bmp", "1280x960.bmp", "2048x1536.bmp"};
    bitmap_image image(images[input_image_size]);
    int h = image.height();
    int w = image.width();
    // RGB to gray
    float *gray = rgb_to_gray(image);
	clock_t start, stop;
	start = clock();
	// No padding: filtered image is smaller than source image
	float *gpu_gray, *gpu_filtered;
	cudaMalloc((void**)&gpu_gray, sizeof(float)*(w*h));
	cudaMalloc((void**)&gpu_filtered, sizeof(float)*((w-2)*(h-2)));
	cudaMemcpy(gpu_gray, gray, sizeof(float)*(w*h), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaCheckErrors("Err 1");
//	start = clock(); // computation only clock
	dim3 block(BSIZE, BSIZE);
	dim3 grid((w + block.x ) / block.x, (h + block.y) / block.y);

	gauss_filter <<< grid, block >>>(gpu_gray, gpu_filtered, w, h);

	cudaDeviceSynchronize();
	cudaCheckErrors("Err 2");
//	stop = clock(); // computation only clock

	float *filtered = new float[(w - 2)*(h - 2)];
	cudaMemcpy(filtered, gpu_filtered, sizeof(float)*(w - 2)*(h - 2),
	cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaFree(gpu_gray);
	cudaFree(gpu_filtered);

	stop = clock();
    cout << "Time passed: " << (stop - start) / (double)CLOCKS_PER_SEC * 1000.0 << " ms";
    bitmap_image G(images[input_image_size]);
    int value;
    for (int y = 0; y < h - 2; y++)
	{
		for (int x = 0; x < w - 2; x++)
		{
			value = (int)(filtered[y*(w-2)+x]);
			G.set_pixel(x+1, y+1, value, value, value);
		}
	}
    G.save_image("output.bmp");
	return 0;
}
