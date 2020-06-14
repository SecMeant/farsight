#include "3d.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <fmt/format.h>

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

class DisjointSet
{
 constexpr static double distanceThreshold = 0.05; // in meters
 constexpr static size_t nan_label = 0;
 constexpr static size_t label_reset = 1;
 constexpr static size_t max_catgory_number = 255;
 
public:
  struct CategoryDescriptor
  {
    static inline size_t next_label = label_reset;
    int size = 0;
    int label = label_reset;
    CategoryDescriptor()
    {
      if(next_label < max_catgory_number)
          label = next_label++;
      else
          label = max_catgory_number;
    }
    CategoryDescriptor(int l)
    {
      label = l;
    }
    CategoryDescriptor(const CategoryDescriptor &c)
    {
        label = c.label;
        size = c.size;
    }
    CategoryDescriptor(CategoryDescriptor &&) = delete;
  };

  struct DisjointPoint
  {
    farsight::Point3f p;
    CategoryDescriptor *category;

    DisjointPoint() =delete;

    DisjointPoint(farsight::Point3f &p1)
      : p(p1)
      , category(nullptr)
    {}
    DisjointPoint(farsight::Point3f &p1, CategoryDescriptor &cd)
      : p(p1)
      , category(&cd)
    {}
  };

  void fixCategory(CategoryDescriptor &from, CategoryDescriptor &to)
  {
    auto f_label = from.label;
    auto t_label = to.label;
    for(auto &cat : categories)
    {
        if(cat.label == f_label)
        {
            assert(cat.label < categories.size());
            cat.label = t_label;
        }
    }
  }

  double
  calcMetric(const farsight::Point3f &p1, const farsight::Point3f &p2)
  {
    assert(!(std::isnan(p1.x) || std::isnan(p2.x)));

    double val = fabs(p1.x - p2.x) + fabs(p1.y - p2.y) + fabs(p1.z - p2.z);
    return val;
  }

  DisjointPoint
  classify(farsight::Point3f &p)
  {
    DisjointPoint p_tmp(p);
    // chech every point in set
    // if point is near enough to some point assign new category
    for (auto &dp : points)
    {
      if (std::isnan(dp.p.x) || calcMetric(dp.p, p) > distanceThreshold)
      {
        continue;
      }

      if (p_tmp.category == nullptr)
      {
        p_tmp.category = dp.category;
        continue;
      }

      if(p_tmp.category->label != dp.category->label)
      {
        fixCategory(*dp.category, *p_tmp.category);
      }
    }
    // if is already attached and can be merged to another group
    // attach group pointer to such group and increment its size
    return p_tmp;
  }

  void
  addPoint(farsight::Point3f p)
  {
    if(likely(std::isnan(p.x)))
    {
      categories[0].size += 1;
      points.emplace_back(p, categories[0]);
      return;
    }
    // if set is empty, create new classification group
    if (unlikely(categories.size() == 1))
    {
      categories.emplace_back();
      categories[1].size = 1;
      points.emplace_back(p, categories[1]);
      return;
    }

    auto classified_p = classify(p);

    if (classified_p.category == nullptr)
    {
        // create new group
        categories.emplace_back();
        auto &cat = categories.back();
        cat.size = 1;
        classified_p.category = &cat;
    }else
    {
        classified_p.category->size += 1;
    }
    points.push_back(classified_p);
  }

  CategoryDescriptor
  findBiggestCategory()
  {
    if (unlikely(categories.size() == 0))
    {
        fmt::print(stderr, "No unions found");
        return {};
    }
    int cat_size = categories.size();
    std::vector<int> categories_lookup(cat_size, 0);
    // sum up all categories
    for(int i =1 ; i < cat_size; i++)
    {
        fmt::print(stderr, "Found category with label {} \n", categories[i].label);
        assert(categories[i].label < cat_size);

        categories_lookup[categories[i].label] += categories[i].size;
    }
    // find biggest category
    size_t max = 0, idx = 0;
    for(int i =1 ; i < cat_size; i++)
    {
        if(max < categories_lookup[i])
        {
            max = categories_lookup[i];
            idx = i;
        }
    }

    for(int i =0 ; i < categories.size(); i++)
    {
        fmt::print("Category with label: {}, size: {}\n", i, categories_lookup[i]);
    }

    return categories[idx];
  }

  farsight::PointArray
  getFilteredPoints(CategoryDescriptor &c1)
  {
    auto label = c1.label;
    farsight::PointArray map;
    int counter = 0;
    for (auto &dp : points)
    {
      auto p_label = dp.category->label;
      if (label == p_label)
      {
        counter++;
        map.push_back(dp.p);
      }else{
        map.push_back({ NAN, NAN, NAN});
      }
    }
    fmt::print("Counter number updated: {}\n", counter);
    return map;
  }

  void reset()
  {
    points.clear();
    categories.clear();
    CategoryDescriptor::next_label = label_reset;
    // add default nan label for nan points
    categories.emplace_back(nan_label);
  }

  DisjointSet()
  {
      // add default nan label for nan points
      categories.emplace_back(nan_label);
  }

private:
  std::vector<DisjointPoint> points;
  std::vector<CategoryDescriptor> categories;
};
