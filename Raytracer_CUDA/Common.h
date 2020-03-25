#pragma once

__device__ const float c_Pi = 3.1415926535897932385;

inline float ffmin(float a, float b) { return a <= b ? a : b; }
inline float ffmax(float a, float b) { return a >= b ? a : b; }

// returns a random real float in (min, max]. Unfortunately curand_uniform returns (0,1]
__device__ inline float randomFloat(float min, float max, curandState* localRandomState)
{
	return min + (max - min) * (curand_uniform(localRandomState));
}