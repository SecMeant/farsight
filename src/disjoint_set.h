#include "3d.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <fmt/format.h>
#include <random>

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)
static thread_local std::mt19937 gen{std::random_device{}()};
 
template<typename T>
T random(T min, T max) {
    return std::uniform_int_distribution<T>{min, max}(gen);
}

class DisjointSet
{
 using CategoryCounter = std::vector<int>;

 double distanceThreshold = 0.02; // in meters
 int objectValidSize = 100; // in meters
 constexpr static size_t nan_label = 0;
 constexpr static size_t point_unset= 424*512;
 constexpr static size_t label_reset = 1;
 constexpr static size_t max_catgory_number = 255;
 
public:
  struct CategoryDescriptor
  {
    static inline size_t next_label = label_reset;
    int size = 0;
    int label = label_reset;
    farsight::ColorType color;
    CategoryDescriptor()
    {
      label = next_label++;
      color.packed = random(0, (1<<24)-1);
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
    size_t category = point_unset;

    DisjointPoint() =delete;

    DisjointPoint(farsight::Point3f &p1)
      : p(p1)
      , category(point_unset)
    {}
    DisjointPoint(farsight::Point3f &p1, size_t cat)
      : p(p1)
      , category(cat)
    {}
  };

  void fixCategory(CategoryDescriptor &from, CategoryDescriptor &to)
  {
    auto f_label = from.label;
    auto t_label = to.label;
    auto color= to.color;
    auto cat_size = categories.size();
    for(auto &cat : categories)
    {
        if(cat.label == f_label)
        {
            cat.label = t_label;
            cat.color= color;
            assert(cat.label < cat_size);
        }
    }
  }

  double
  calcSquareMetric(const farsight::Point3f &p1, const farsight::Point3f &p2)
  {
    assert(!(std::isnan(p1.x) || std::isnan(p2.x)));

    double val = sqrt(pow((p1.x - p2.x),2) + pow((p1.y - p2.y),2) +pow((p1.z - p2.z),2));
    return val;
  }

  void
  updateTreshold(double tr)
  {
    fmt::print("Current treshold: {}\n", tr);
    distanceThreshold = tr;
  }

  void
  updateValidSize(double size)
  {
    fmt::print("Size for valid object: {}\n", size);
    objectValidSize = size;
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

      if (p_tmp.category == point_unset)
      {
        p_tmp.category = dp.category;
        continue;
      }

      if(categories[p_tmp.category].label != categories[dp.category].label)
      {
        fixCategory(categories[dp.category], categories[p_tmp.category]);
      }
    }
    // if is already attached and can be merged to another group
    // attach group pointer to such group and increment its size
    return p_tmp;
  }

  void
  addPoint(farsight::Point3f p)
  {
    if(unlikely(categories.size() == 0))
    {
      // add default nan label for nan points
      categories.emplace_back(nan_label);
    }

    if(likely(std::isnan(p.x)))
    {
      categories[0].size += 1;
      points.emplace_back(p, 0);
      return;
    }
    // if set is empty, create new classification group
    if (unlikely(categories.size() == 1))
    {
      categories.emplace_back();
      categories[1].size = 1;
      points.emplace_back(p, 1);
      return;
    }

    auto classified_p = classify(p);

    if(classified_p.category == point_unset)
    {
        // create new group
        categories.emplace_back();
        auto &cat = categories.back();
        cat.size = 1;
        classified_p.category = categories.size()-1;
    }else
    {
        categories[classified_p.category].size += 1;
    }
    points.push_back(classified_p);
  }

  CategoryCounter
  countCategories()
  {
    if(unlikely(categories.size() == 0))
    {
        fmt::print(stderr, "No unions found");
        return {};
    }
    int cat_size = categories.size();
    CategoryCounter categories_lookup = CategoryCounter(cat_size, 0);
    // sum up all categories
    for(int i =1 ; i < cat_size; i++)
    {
        assert(categories[i].label < cat_size);

        categories_lookup[categories[i].label] += categories[i].size;
    }

    return categories_lookup;
  }

  CategoryDescriptor
  findBiggestCategory(CategoryCounter &cc)
  {
    int cat_size = categories.size();
    // find biggest category
    size_t max = 0, idx = 0;
    for(int i =1 ; i < cat_size; i++)
    {
        if(max <cc[i])
        {
            max =cc[i];
            idx = i;
        }
    }

    for(int i =0 ; i < categories.size(); i++)
    {
        fmt::print("Category with label: {}, size: {}\n", i,cc[i]);
    }

    return categories[idx];
  }

  farsight::PointArray
  getFilteredPoints(CategoryDescriptor &c1)
  {
    auto label = c1.label;
    farsight::PointArray map;
    farsight::ColorType color;
    farsight::Point3f nan_p = { NAN, NAN, NAN};
    for (auto &dp : points)
    {
      auto p_label = categories[dp.category].label;
      if (label == p_label)
      {
        color.packed = 0xdd88ff;
        map.emplace_back(dp.p, color);
      }else{
        color.packed = 0x0;
        map.emplace_back(nan_p, color);
      }
    }
    return map;
  }

  farsight::PointArray
  getPointsByDelimiter(CategoryCounter &cc)
  {
    farsight::PointArray map;
    farsight::ColorType color;
    farsight::Point3f nan_p = { NAN, NAN, NAN};
    for (auto &dp : points)
    {
      auto label= categories[dp.category].label;
      if (cc[label] > objectValidSize)
      {
        color.packed = 0xdd88ff;
        map.emplace_back(dp.p, color);
      }else{
        color.packed = 0x0;
        map.emplace_back(nan_p, color);
      }
    }
    return map;
  }

  farsight::PointArray
  getFilteredPointsColors(CategoryDescriptor &c1)
  {
    auto label = c1.label;
    farsight::PointArray map;
    int counter = 0;
    for (auto &dp : points)
    {
      auto color = categories[dp.category].color;
      map.emplace_back(dp.p, color);
    }
    return map;
  }

  void reset()
  {
    points.clear();
    categories.clear();
    CategoryDescriptor::next_label = label_reset;
  }

private:
  std::vector<DisjointPoint> points;
  std::vector<CategoryDescriptor> categories;
};
