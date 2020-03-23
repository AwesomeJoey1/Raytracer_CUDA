#pragma once
#include "Ray.h"

class Material;

struct hitRecord {
	float t;
	glm::vec3 p;
	glm::vec3 normal;
	Material *material;
};

class Hittable
{
public:
	__device__ virtual bool hit(const Ray& ray, float tMin, float tMax, hitRecord &rec) const = 0;
};