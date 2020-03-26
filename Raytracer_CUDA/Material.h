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

__device__ glm::vec3 randomUnitVector(curandState* localRandState)
{
	float a = randomFloat(0, 2 * c_Pi, localRandState);
	float z = randomFloat(-1, 1, localRandState);
	float r = glm::sqrt(1 - z * z);

	return glm::vec3(r * cos(a), r * sin(a), z);
}

__device__ glm::vec3 reflect(glm::vec3& vecIn, glm::vec3& normal)
{
	return vecIn - 2.0f * glm::dot(vecIn, normal) * normal;
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
		glm::vec3 target = rec.p + rec.normal + randomUnitVector(localRandState);
		rayScattered = Ray(rec.p, target - rec.p);
		attenuation = _albedo;
		return true;
	}

private:
	glm::vec3 _albedo;
};

class Metal : public Material
{
public:
	__device__ Metal(const glm::vec3& albedo, const float fuzz) : _albedo(albedo), _fuzz(fuzz < 1.0f ? fuzz : 1.0f) { }
	__device__ virtual bool scatter(const Ray& ray, const hitRecord& rec, glm::vec3& attenuation, Ray& rayScattered, curandState* localRandState) const
	{
		glm::vec3 reflected = reflect(glm::normalize(ray.direction()), rec.normal);
		rayScattered = Ray(rec.p, reflected + _fuzz * randomInUnitSphere(localRandState));
		attenuation = _albedo;

		return (glm::dot(rayScattered.direction(), rec.normal) > 0.0f);
	}

private:
	glm::vec3 _albedo;
	float _fuzz;
};