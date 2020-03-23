#pragma once
#include "Hittable.h"

class Sphere : public Hittable
{
public:
	__device__ Sphere () {}
	__device__ Sphere(glm::vec3 center, float radius) : _center(center), _radius(radius) {};

	__device__ virtual bool hit(const Ray& ray, float tMin, float tMax, hitRecord& rec) const;


private:
	glm::vec3 _center;
	float _radius;
};

__device__ bool Sphere::hit(const Ray& ray, float tMin, float tMax, hitRecord& rec) const
{
    glm::vec3 oc = ray.origin() - _center;
    float a = glm::dot(ray.direction(), ray.direction());
    float b = glm::dot(oc, ray.direction());
    float c = glm::dot(oc, oc) - _radius * _radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < tMax && temp > tMin) {
            rec.t = temp;
            rec.p = ray.pointAt(rec.t);
            rec.normal = (rec.p - _center) / _radius;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < tMax && temp > tMin) {
            rec.t = temp;
            rec.p = ray.pointAt(rec.t);
            rec.normal = (rec.p - _center) / _radius;
            return true;
        }
    }
    return false;
}
