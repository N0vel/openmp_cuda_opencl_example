#include <iostream>
#include "BMP.h"
#include <cmath>
#include <omp.h>
#include <ctime>

using namespace std;

int main() {
	char images [3][50] = {"1024x768.bmp", "1280x960.bmp", "2048x1536.bmp"};
	clock_t start, stop;

	for (int v=0; v<3; v++)
	{
		BMP bmp(images[v]);
		BMP filtered_image = bmp;
		vector<int> number_of_threads = {2, 4, 6, 8, 10, 12, 16};
		int n_threads = 0;
		for (int i=0; i < 7; i++)
		{
			double mean_time = 0.;
			for (int j=0; j<3; j++)
			{
				n_threads = number_of_threads[i];
				omp_set_num_threads( n_threads );
				filtered_image = bmp;
				start = clock();
				#pragma omp parallel
				{
					int threadnum = omp_get_thread_num(), numthreads = omp_get_num_threads();
					int low = filtered_image.bmp_info_header.height*threadnum/numthreads, high = filtered_image.bmp_info_header.height*(threadnum+1)/numthreads;
					// grayscale
					double gray_value = 0.;
					uint32_t channels = filtered_image.bmp_info_header.bit_count / 8;
					for (uint32_t y = low; y < high; ++y) {
						for (uint32_t x = 0; x < filtered_image.bmp_info_header.width; ++x) {
							gray_value = filtered_image.data[channels * (y * filtered_image.bmp_info_header.width + x) + 0] +
							filtered_image.data[channels * (y * filtered_image.bmp_info_header.width + x) + 1] +
							filtered_image.data[channels * (y * filtered_image.bmp_info_header.width + x) + 2];
							gray_value = uint8_t(round(gray_value / 3.));
							filtered_image.data[channels * (y * filtered_image.bmp_info_header.width + x) + 0] = gray_value;
							filtered_image.data[channels * (y * filtered_image.bmp_info_header.width + x) + 1] = gray_value;
							filtered_image.data[channels * (y * filtered_image.bmp_info_header.width + x) + 2] = gray_value;
						}
					}
					// filtering
					vector<double> g_filter = {1./16., 2./16., 1./16., 2./16., 4./16., 2./16., 1./16., 2./16., 1./16.};
					double gauss_value = 0.;
					for (uint32_t y = low; y < high; y++) 
					{
						for (uint32_t x = 0; x < bmp.bmp_info_header.width; x++) 
						{
							if ((y > 0) && (y < bmp.bmp_info_header.height-1) && (x > 0) && (x < bmp.bmp_info_header.width-1))
							{
								for (int i = 0; i < 3; i++)
								{
									for (int j = 0; j < 3; j++)
									{
										gauss_value += g_filter[3*i+j]*double(bmp.data[channels * ((y-1+i) * bmp.bmp_info_header.width + x-1+j) + 0]);
									}
								}
							}
							filtered_image.data[channels * (y * filtered_image.bmp_info_header.width + x) + 0] = uint8_t(round(gauss_value));
							filtered_image.data[channels * (y * filtered_image.bmp_info_header.width + x) + 1] = uint8_t(round(gauss_value));
							filtered_image.data[channels * (y * filtered_image.bmp_info_header.width + x) + 2] = uint8_t(round(gauss_value));
							gauss_value = 0.;
						}
					}
				}
				stop = clock();
				filtered_image.write(("gauss_filtered" + to_string(v)+".bmp").c_str());
				mean_time += (stop - start) / (double)CLOCKS_PER_SEC * 1000.0 /3.;
			}
			cout << n_threads << " threads. " << "It took " << mean_time << "ms" << std::endl;
			mean_time = 0.;
		}
	}
	return 0;
}