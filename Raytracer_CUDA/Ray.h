#pragma once
#include <glm/glm/glm.hpp>

class Ray
{
public:
	__device__ Ray() {}
	__device__ Ray(glm::vec3 &origin, glm::vec3 &direction) { _origin = origin; _direction = direction; }
	__device__ glm::vec3 origin() const { return _origin; }
	__device__ glm::vec3 direction() const { return _direction; }
	__device__ glm::vec3 pointAt(float t) const { return _origin + t * _direction; }

private:
	glm::vec3 _origin, _direction;
};
