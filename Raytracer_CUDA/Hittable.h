#pragma once
#include "Ray.h"

struct hitRecord {
	glm::vec3 p;
	float t;
	glm::vec3 normal;
};

class Hittable
{
public:
	__device__ virtual bool hit(const Ray& ray, float tMin, float tMax, hitRecord &rec) const = 0;
};