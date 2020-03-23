#pragma once
#include "Hittable.h"


class HittableList : public Hittable
{
public:
	__device__ HittableList() {}
	__device__ HittableList(Hittable** list, int n) { _list = list, _listSize = n; }

	__device__ virtual bool hit(const Ray &ray, float tMin, float tMax, hitRecord &rec) const;

private:
	Hittable **_list;
	int _listSize;
};

__device__ bool HittableList::hit(const Ray& ray, float tMin, float tMax, hitRecord& rec) const
{
	hitRecord hitRecTemp;
	bool hitAnything = false;
	float closestSoFar = tMax;

	for (int i = 0; i < _listSize; i++)
	{
		if (_list[i]->hit(ray, tMin, closestSoFar, hitRecTemp))
		{
			hitAnything = true;
			closestSoFar = rec.t;
			rec = hitRecTemp;
		}
	}
	return hitAnything;
}