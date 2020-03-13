#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h> // only that IntelliJ doesn't complain about threadIdx and similar stuff

#define GLM_FORCE_CUDA
#include <glm/glm/glm.hpp>
#include <iostream>
#include <time.h>

#include "Image.h"
#include "Ray.h"

#define IMG_WIDTH 200
#define IMG_HEIGHT 100
#define TX 8
#define TY 8

// checks cuda function calls for erroneous behavior
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

// calculates the color of one ray
 __device__ glm::vec3 color(const Ray &ray)
{
	glm::vec3 unitDirection = glm::normalize(ray.direction());
	float t = 0.5f * (unitDirection.y + 1.0f);
	return (1.0f - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
}

 // cuda function to render full scene into a framebuffer
__global__ void render(glm::vec3* frameBuffer, int width, int height, glm::vec3 lowerLeftCorner, glm::vec3 horizontal, glm::vec3 vertical, glm::vec3 origin)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	// Image might not be multiple of the kernel blocks. 
	// Threads that are not supposed to write to the image buffer will be sorted out
	if ((i >= width) || (j >= height)) return;

	int pixelIndex = j * width + i;
	float u = float(i) / float(width);
	float v = float(j) / float(height);
	frameBuffer[pixelIndex] = color(Ray(origin, lowerLeftCorner + u * horizontal + v * vertical));
}

int main()
{
	Image image(IMG_WIDTH, IMG_HEIGHT);
	clock_t start, stop, end;

	int numPixels = IMG_WIDTH * IMG_HEIGHT;
	size_t frameBufferSize = numPixels * sizeof(glm::vec3);

	std::cout << "Allocating memory...";
	start = clock();
	glm::vec3* frameBuffer;
	checkCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));
	stop = clock();
	std::cout << "\t\t" << (stop - start) << "ms\n";
	

	dim3 blocks(IMG_WIDTH / TX + 1, IMG_HEIGHT / TY + 1);
	dim3 threads(TX, TY);
	std::cout << "Rendering...";
	stop = clock();

	render<<<blocks, threads>>>(frameBuffer, IMG_WIDTH, IMG_HEIGHT, 
		glm::vec3(-2.0f, -1.0f, -1.0f),	// lower left corner
		glm::vec3(4.0f, 0.0f, 0.0f),	// horizontal 
		glm::vec3(0.0f, 2.0f, 0.0f),	// vertical
		glm::vec3(0.0f,0.0f,0.0f));		// origin

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	end = clock();
	std::cout << "\t\t\t" << (end - stop) << "ms\n";
	std::cout << "Total:\t\t\t\t" << (end - start) << "ms\n";

	for (int j = IMG_HEIGHT - 1; j >= 0; j--) {
		for (int i = 0; i < IMG_WIDTH; i++) {
			size_t pixelIndex = j * IMG_WIDTH + i;

			glm::vec3 pixelColor = frameBuffer[pixelIndex];
			pixelColor *= 255.99f;

			image.setPixel(i, j, pixelColor);
		}
	}

	std::cout << "Saving Image!\n";
	image.save();
	std::cout << "Done!\n";

	checkCudaErrors(cudaFree(frameBuffer));

	return 0;
}