#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h> // only that IntelliJ doesn't complain about threadIdx and similar stuff

#define GLM_FORCE_CUDA
#include <glm/glm/glm.hpp>
#include <iostream>
#include <time.h>

#include "Image.h"

#define IMG_WIDTH 200
#define IMG_HEIGHT 100
#define TX 8
#define TY 8

#define checkCudaErrors(val) checkCuda( (val), #val, __FILE__, __LINE__)
void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";

		cudaDeviceReset();
		exit(99);
	}
}

__global__ void render(float* frameBuffer, int width, int height)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	// Image might not be multiple of the kernel blocks. 
	// Threads that are not supposed to write to the image buffer will be sorted out
	if ((i >= width) || (j >= height)) return;
	int pixelIndex = j * width * 3 + i * 3;
	frameBuffer[pixelIndex] = float(i) / width;
	frameBuffer[++pixelIndex] = float(j) / height;
	frameBuffer[++pixelIndex] = 0.2;
}

int main()
{
	Image image(IMG_WIDTH, IMG_HEIGHT);
	clock_t start, stop, end;

	int numPixels = IMG_WIDTH * IMG_HEIGHT;
	size_t frameBufferSize = 3 * numPixels * sizeof(float);

	std::cout << "Allocating memory...";
	start = clock();
	float* frameBuffer;
	checkCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));
	stop = clock();
	std::cout << "\t\t" << (stop - start) << "ms\n";
	

	dim3 blocks(IMG_WIDTH / TX + 1, IMG_HEIGHT / TY + 1);
	dim3 threads(TX, TY);
	std::cout << "Rendering...";
	stop = clock();
	render<<<blocks, threads>>>(frameBuffer, IMG_WIDTH, IMG_HEIGHT);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	end = clock();
	std::cout << "\t\t\t" << (end - stop) << "ms\n";
	std::cout << "Total:\t\t\t\t" << (end - start) << "ms\n";

	for (int j = IMG_HEIGHT - 1; j >= 0; j--) {
		for (int i = 0; i < IMG_WIDTH; i++) {
			size_t pixelIndex = j * 3 * IMG_WIDTH + i * 3;

			float r = frameBuffer[pixelIndex];
			float g = frameBuffer[pixelIndex + 1];
			float b = frameBuffer[pixelIndex + 2];
			int ir = r * 255.99;
			int ig = g * 255.99;
			int ib = b * 255.99;

			image.setPixel(i, j, glm::vec3(ir, ig, ib));
		}
	}

	std::cout << "Saving Image!\n";
	image.save();
	std::cout << "Done!\n";

	checkCudaErrors(cudaFree(frameBuffer));

	return 0;
}