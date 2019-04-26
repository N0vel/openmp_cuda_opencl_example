#include "cuda_runtime.h"
#include "debice_launch_parameters"
#include <iostream>
#include "BMP.h"
#include <chrono>

using namespace std;

#define BSIZE 32
#define input_image_size 0

__global__  to_grayscale(BMP filtered_image)
{
    int low = filtered_image.bmp_info_header.height*threadnum/numthreads, high = filtered_image.bmp_info_header.height*(threadnum+1)/numthreads;
    double gray_value = 0.;
    uint32_t channels = filtered_image.bmp_info_header.bit_count / 8;
    
    return filtered_image
}





int main() 
{
	using clock = std::chrono::system_clock;
	using ms = std::chrono::milliseconds;
    char images [3][50] = {"1024x768.bmp", "1280x960.bmp", "2048x1536.bmp"};
    bitmap_image bmp(images[input_image_size]);
    unsigned int h = bmp.height()
    unsigned int w = bmp.width()

    const auto before = clock::now();

    const auto duration = std::chrono::duration_cast<ms>(clock::now() - before);
    cout << "Time passed: " << duration / 1000. << " ms"
	return 0;
}