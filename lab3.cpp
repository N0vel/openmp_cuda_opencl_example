#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif 
#include <iostream>
#include "bitmap_image.hpp"
#include <ctime>
#include <stdio.h>
using namespace std;

#define MAX_SOURCE_SIZE (1048576) //1 MB
#define MAX_LOG_SIZE    (1048576) //1 MB


float* rgb_to_gray(bitmap_image image)
{
	int h = image.height();
	int w = image.width();
    unsigned char r, g, b;
	float *gray = new float[w*h*3];
    float value;
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			image.get_pixel(x, y, r, g, b);
            value = (float) ((r+g+b) / 3.);
			gray[(y*w+x)*3] = value;
            gray[(y*w+x)*3+1] = value;
            gray[(y*w+x)*3+2] = value;
		}

	}
	return gray;
}

int main() 
{
    char images [3][50] = {"1024x768.bmp", "1280x960.bmp", "2048x1536.bmp"};
    cl_int ret;//the openCL error code/s
    int input_image_size = 0;
    int group_size = 1;


    bitmap_image image(images[input_image_size]);
    int h = image.height();
    int w = image.width();
    float G[9] = {1./16., 2./16., 1./16., 2./16., 4./16., 2./16., 1./16., 2./16., 1./16.};
    // RGB to gray
    float *gray = rgb_to_gray(image);
    int imgSize = w*h*3;
    float *filtered = new float[imgSize];
    int size = 3;
	clock_t start, stop;
	start = clock();

    // Read in the kernel code into a c string
    FILE* f;
    char* kernelSource;
    size_t kernelSrcSize;
    if( (f = fopen("gauss_kernel.cl", "r")) == NULL)
    {
        fprintf(stderr, "Failed to load OpenCL kernel code.\n");
        return false;
    }
    kernelSource = (char *)malloc(MAX_SOURCE_SIZE);
    kernelSrcSize = fread( kernelSource, 1, MAX_SOURCE_SIZE, f);
    fclose(f);

    //Get platform and device information
    cl_platform_id platformID; //will hold the ID of the openCL available platform
    cl_uint platformsN; //will hold the number of openCL available platforms on the machine
    cl_device_id deviceID; //will hold the ID of the openCL device
    cl_uint devicesN; //will hold the number of OpenCL devices in the system
    if(clGetPlatformIDs(1, &platformID, &platformsN) != CL_SUCCESS)
    {
        printf("Could not get the OpenCL Platform IDs\n");
        return false;
    }
    if(clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1,&deviceID, &devicesN) != CL_SUCCESS)
    {
        printf("Could not get the system's OpenCL device\n");
        return false;
    }
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &deviceID, NULL, NULL, &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Could not create a valid OpenCL context\n");
        return false;
    }
    // Create a command queue
    cl_command_queue cmdQueue = clCreateCommandQueue(context, deviceID, 0, &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Could not create an OpenCL Command Queue\n");
        return false;
    }

    // Create memory buffers on the device for the two images
    cl_mem gpuImg = clCreateBuffer(context,CL_MEM_READ_ONLY,imgSize*sizeof(float), NULL,&ret);
    if(ret != CL_SUCCESS)
    {
        printf("Unable to create the GPU image buffer object\n");
        return false;
    }
    cl_mem gpuGaussian = clCreateBuffer(context, CL_MEM_READ_ONLY, size*size*sizeof(float), NULL, &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Unable to create the GPU image buffer object\n");
        return false;
    }
    cl_mem gpuNewImg = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imgSize*sizeof(float), NULL, &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Unable to create the GPU image buffer object\n");
        return false;
    }

    //Copy the image data and the gaussian kernel to the memory buffer
    if(clEnqueueWriteBuffer(cmdQueue, gpuImg, CL_TRUE, 0, imgSize*sizeof(float), gray, 0, NULL, NULL) != CL_SUCCESS)
    {
        printf("Error during sending the image data to the OpenCL buffer\n");
        return false;
    }
    if(clEnqueueWriteBuffer(cmdQueue, gpuGaussian, CL_TRUE, 0,size*size*sizeof(float), G, 0, NULL, NULL) != CL_SUCCESS)
    {
        printf("Error during sending the gaussian kernel to the OpenCL buffer\n");
        return false;
    }

    //Create a program object and associate it with the kernel's source code.
    cl_program program = clCreateProgramWithSource(context, 1,(const char **)&kernelSource, (const size_t *)&kernelSrcSize, &ret);
    free(kernelSource);
    if(ret != CL_SUCCESS)
    {
        printf("Error in creating an OpenCL program object\n");
        return false;
    }
    //Build the created OpenCL program
    if((ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL))!= CL_SUCCESS)
    {
        printf("Failed to build the OpenCL program\n");
        //create the log string and show it to the user. Then quit
        char* buildLog;
        buildLog = (char *) malloc(MAX_LOG_SIZE);
        
        if(clGetProgramBuildInfo(program,deviceID,CL_PROGRAM_BUILD_LOG,MAX_LOG_SIZE,buildLog,NULL) != CL_SUCCESS)
        {
            printf("Could not get any Build info from OpenCL\n");
            free(buildLog);
            return false;
        }
        printf("**BUILD LOG**\n%s",buildLog);
        free(buildLog);
        return false;
    }
    // Create the OpenCL kernel. This is basically one function of the program declared with the __kernel qualifier
    cl_kernel kernel = clCreateKernel(program, "gaussian_blur", &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Failed to create the OpenCL Kernel from the built program\n");
        return false;
    }
    ///Set the arguments of the kernel
    if(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&gpuImg) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"gpuImg\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&gpuGaussian) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"gpuGaussian\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 2, sizeof(int), (void *)&w) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"imageWidth\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 3, sizeof(int), (void *)&h) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"imgHeight\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&gpuNewImg) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"gpuNewImg\" argument\n");
        return false;
    }

    // enqueue the kernel into the OpenCL device for execution
    size_t globalWorkItemSize = imgSize;//the total size of 1 dimension of the work items. Basically the whole image buffer size
    size_t workGroupSize = group_size; //The size of one work group
    ret = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, &globalWorkItemSize, &workGroupSize,0, NULL, NULL);

    //Read the memory buffer of the new image on the device to the new Data local variable
    ret = clEnqueueReadBuffer(cmdQueue, gpuNewImg, CL_TRUE, 0,imgSize*sizeof(float), filtered, 0, NULL, NULL);

    // free(G);
    clFlush(cmdQueue);
    clFinish(cmdQueue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(gpuImg);
    clReleaseMemObject(gpuGaussian);
    clReleaseMemObject(gpuNewImg);
    clReleaseCommandQueue(cmdQueue);
    clReleaseContext(context);

	stop = clock();
    cout << "Time passed: " << (stop - start) / (double)CLOCKS_PER_SEC * 1000.0 << " ms";
    bitmap_image output(images[input_image_size]);
    int value;
    for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			value = (int)(filtered[3*(y*w+x)]);
			output.set_pixel(x, y, value, value, value);
		}
	}
    output.save_image("output.bmp");
	return 0;
}
