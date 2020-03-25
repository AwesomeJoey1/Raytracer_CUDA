#pragma once
#include "Ray.h"

class Material;

struct hitRecord {
	float t;
	glm::vec3 p;
	glm::vec3 normal;
	bool frontFace;
	Material *material;

	__device__ inline void setFaceNormal(const Ray& ray, const glm::vec3& outwardNormal)
	{
		frontFace = glm::dot(ray.direction(), outwardNormal) < 0;
		normal = frontFace ? outwardNormal : -outwardNormal;
	}
};

class Hittable
{
public:
	__device__ virtual bool hit(const Ray& ray, float tMin, float tMax, hitRecord &rec) const = 0;
};