#pragma once

struct hitRecord;

#include "Ray.h"
#include "Hittable.h"

#define RANDVEC3 glm::vec3(curand_uniform(localRandState),curand_uniform(localRandState),curand_uniform(localRandState))

__device__ glm::vec3 randomInUnitSphere(curandState* localRandState)
{
	glm::vec3 randomPoint;
	do
	{
		randomPoint = 2.0f * RANDVEC3 - glm::vec3(1.0f, 1.0f, 1.0f);
	} while (glm::length2(randomPoint) >= 1.0f);
	
	return randomPoint;
}

class Material
{
public:
	__device__ virtual bool scatter(const Ray& ray, const hitRecord& rec, glm::vec3& attenuation, Ray& rayScattered, curandState* localRandState) const = 0;

};

class Lambertian : public Material
{
public:
	__device__ Lambertian(const glm::vec3& albedo) : _albedo(albedo) {}
	__device__ virtual bool scatter(const Ray& ray, const hitRecord& rec, glm::vec3& attenuation, Ray& rayScattered, curandState* localRandState) const
	{
		glm::vec3 target = rec.p + rec.normal + randomInUnitSphere(localRandState);
		rayScattered = Ray(rec.p, target - rec.p);
		attenuation = _albedo;
		return true;
	}

private:
	glm::vec3 _albedo;
};