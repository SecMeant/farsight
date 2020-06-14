#include "disjoint_set.h"
#include <vector>
#include "3d.h"

std::vector<farsight::Point3f> points = {{0.01, 0.0, 0.2},
{0.01, 0.0, 0.2},{0.01, 0.0, 0.2},{0.01, 0.0, 0.2},{0.01, 0.0, 0.2},{0.01, 0.0, 0.2},{0.01, 0.0, 0.2},{0.01, 0.0, 0.2},{0.01, 0.0, 0.2},{0.01, 0.0, 0.2},{0.01, 0.0, 0.2},{0.01, 0.0, 0.2},{0.01, 0.0, 0.2},{0.01, 0.0, 0.2},{0.01, 0.0, 0.2},{0.01, 0.0, 0.2}
};

DisjointSet classifier;

int main()
{
    for(auto &p : points)
    {
        classifier.addPoint(p);
    }
    auto cat = classifier.findBiggestCategory();
    classifier.getFilteredPoints(cat);
}
