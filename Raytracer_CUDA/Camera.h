#pragma once
#include "Ray.h"

class Camera
{
public:
	__device__ Camera(glm::vec3 lowerLeft, glm::vec3 horizontal, glm::vec3 vertical, glm::vec3 origin) : 
		_lowerLeft(lowerLeft), _horizontal(horizontal), _vertical(vertical), _origin(origin) {}

	__device__ Ray getRay(float u, float v) { return Ray(_origin, _lowerLeft + u*_horizontal + v * _vertical - _origin); }

private:
	glm::vec3 _lowerLeft, _horizontal, _vertical, _origin;
};
