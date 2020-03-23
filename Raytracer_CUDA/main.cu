#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h> // only that IntelliJ doesn't complain about threadIdx and similar stuff

#define GLM_FORCE_CUDA
#include <glm/glm/glm.hpp>
#include <glm/glm/ext/scalar_constants.hpp>
#include <iostream>
#include <time.h>
#include <float.h>
#include <math.h>

#include "Image.h"
#include "Ray.h"
#include "Camera.h"
#include "HittableList.h"
#include "Sphere.h"

#define IMG_WIDTH 200
#define IMG_HEIGHT 100
#define NUM_SAMPLES 10
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
 __device__ glm::vec3 color(const Ray &ray, Hittable **world)
{

	hitRecord rec;
	if ((*world)->hit(ray, 0.0, FLT_MAX, rec)) {
		glm::vec3 normalColor = 0.5f * glm::vec3(rec.normal.x + 1.0f, rec.normal.y + 1.0f, rec.normal.z + 1.0f);
		
		//glm::vec2 texCoordinate;
		// In this particular case, the normal is similar to a point on a unit sphere
		// centred around the origin. We can thus use the normal coordinates to compute
		// the spherical coordinates of Phit.
		// atan2 returns a value in the range [-pi, pi] and we need to remap it to range [0, 1]
		// acosf returns a value in the range [0, pi] and we also need to remap it to the range [0, 1]
		//texCoordinate.x = (1 + atan2(rec.normal.z, rec.normal.x) / glm::pi<float>()) * 0.5;
		//texCoordinate.y = acosf(rec.normal.y) / glm::pi<float>();

		//float scale = 4;
		//float pattern = (fmodf(texCoordinate.x * scale, 1) > 0.5) ^ (fmodf(texCoordinate.y * scale, 1) > 0.5);

		//return glm::max(0.f, glm::dot(rec.normal, -glm::normalize(ray.direction()))) * glm::mix(normalColor, normalColor * 0.8f, pattern);
		return normalColor;
	}
	else {
		glm::vec3 direction = glm::normalize(ray.direction());
		float t = 0.5f * (direction.y + 1.0f);
		return (1.0f - t) * glm::vec3(1.0, 1.0, 1.0) + t * glm::vec3(0.5, 0.7, 1.0);
	}
}

 __global__ void renderInit(int width, int height, curandState* randState)
 {
	 int i = threadIdx.x + blockIdx.x * blockDim.x;
	 int j = threadIdx.y + blockIdx.y * blockDim.y;

	 if (i >= width || j >= height) return;

	 int pixelIndex = j * width + i;
	 // Each thread gets same seed, different sequence number, no offset
	 curand_init(1984, pixelIndex, 0, &randState[pixelIndex]);
 }

 // cuda function to render full scene into a framebuffer
__global__ void render(glm::vec3* frameBuffer, int width, int height, int numSamples, Camera **camera, Hittable **world, curandState *randState)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	// Image might not be multiple of the kernel blocks. 
	// Threads that are not supposed to write to the image buffer will be sorted out
	if ((i >= width) || (j >= height)) return;

	int pixelIndex = j * width + i;
	curandState localRandState = randState[pixelIndex];
	glm::vec3 col(0.0f, 0.0f, 0.0f);
	for (int s = 0; s < numSamples; s++)
	{
		float u = float(i + curand_uniform(&localRandState)) / float(width);
		float v = float(j + curand_uniform(&localRandState)) / float(height);
		Ray ray = (*camera)->getRay(u, v);
		col += color(ray, world);
	}
	frameBuffer[pixelIndex] = col / float(numSamples);
}

__global__ void createWorld(Hittable **_dList, Hittable **_dWorld, Camera **_dCamera)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		*(_dList) = new Sphere(glm::vec3(0.0f, 0.0f, -1.0f), 0.5f);
		*(_dList + 1) = new Sphere(glm::vec3(0.0f, -100.5f, -1.0f), 100.0f);
		*_dWorld = new HittableList(_dList, 2);
		*_dCamera = new Camera(glm::vec3(-2.0f, -1.0f, -1.0f),	// lower left corner
			glm::vec3(4.0f, 0.0f, 0.0f),	// horizontal 
			glm::vec3(0.0f, 2.0f, 0.0f),	// vertical
			glm::vec3(0.0f, 0.0f, 0.0f));	// origin
	}
}

__global__ void freeWorld(Hittable **_dList, Hittable **_dWorld, Camera **_dCamera)
{
	delete *(_dList);
	delete *(_dList + 1);
	delete *_dWorld;
	delete* _dCamera;
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
	

	Hittable **_dList;
	checkCudaErrors(cudaMalloc((void**) &_dList, 2 * sizeof(Hittable*)));
	Hittable **_dWorld;
	checkCudaErrors(cudaMalloc((void**) &_dWorld, sizeof(Hittable*)));
	Camera **_dCamera;
	checkCudaErrors(cudaMalloc((void**)&_dCamera, sizeof(Camera*)));
	createWorld<<<1,1>>>(_dList, _dWorld, _dCamera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// we need to remember the state of every thread on the GPU for random numbers
	curandState* _dRandState;
	checkCudaErrors(cudaMalloc((void**)&_dRandState, numPixels * sizeof(curandState))); 

	
	dim3 blocks(IMG_WIDTH / TX + 1, IMG_HEIGHT / TY + 1);
	dim3 threads(TX, TY);

	renderInit<<<blocks, threads>>>(IMG_WIDTH, IMG_HEIGHT, _dRandState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "Rendering...";
	stop = clock();
	render<<<blocks, threads>>>(frameBuffer, IMG_WIDTH, IMG_HEIGHT,  NUM_SAMPLES, _dCamera, _dWorld, _dRandState);

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

	checkCudaErrors(cudaDeviceSynchronize());
	freeWorld<<<1,1>>>(_dList, _dWorld, _dCamera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(_dList));
	checkCudaErrors(cudaFree(_dWorld));
	checkCudaErrors(cudaFree(_dCamera));
	checkCudaErrors(cudaFree(frameBuffer));

	cudaDeviceReset();

	return 0;
}