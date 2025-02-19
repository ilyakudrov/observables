#pragma once

#include "jackknife.h"

#include <iostream>
#include <unordered_map>
#include <vector>

class JackknifeVisitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::tuple<double, double> result_type;

  JackknifeVisitor(unsigned long bin_size) : bin_size_(bin_size) {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end) {
    std::vector<unsigned long> bin_borders =
        get_bin_borders(idx_end - idx_begin, bin_size_);
    std::vector<double> jackknife_vec =
        do_jackknife(value1_begin, value1_end, bin_borders);
    result_ = get_aver(jackknife_vec);
  }
  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  unsigned long bin_size_;
  result_type result_;
};

template <class T> class RadiusSquaredVisitor {
public:
  typedef unsigned long index_type;
  typedef T value_type;
  typedef std::vector<T> result_type;

  RadiusSquaredVisitor(const int shift) : shift_(shift) {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end, const H &value2_begin,
                  const H &value2_end) {
    result_ = result_type(idx_end - idx_begin);
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    auto it2 = value2_begin;
    for (; it1 != value1_end && it2 != value2_end; ++it1, ++it2, ++it_result) {
      *it_result =
          (*it1 - shift_) * (*it1 - shift_) + (*it2 - shift_) * (*it2 - shift_);
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  int shift_;
  result_type result_;
};

class JvVisitor1 {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  JvVisitor1() {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void
  operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
             const H &value1_end, const H &value2_begin, const H &value2_end,
             const H &value3_begin, const H &value3_end, const H &value4_begin,
             const H &value4_end, const H &value5_begin, const H &value5_end) {
    result_ = result_type(idx_end - idx_begin);
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    auto it2 = value2_begin;
    auto it3 = value3_begin;
    auto it4 = value4_begin;
    auto it5 = value5_begin;
    int count = 0;
    for (; it1 != value1_end && it2 != value2_end && it3 != value3_end &&
           it4 != value4_end && it5 != value5_end;
         ++it1, ++it2, ++it3, ++it4, ++it5, ++it_result) {
      count++;
      *it_result = *it1 + *it2 + *it3 + *it4 - *it5;
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  result_type result_;
};

class JvVisitor2 {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  JvVisitor2(result_type &result) : result_(std::move(result)) {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end, const H &value2_begin,
                  const H &value2_end) {
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    auto it2 = value2_begin;
    int count = 0;
    for (; it1 != value1_end && it2 != value2_end; ++it1, ++it2, ++it_result) {
      count++;
      *it_result = -2 * (*it_result - *it1 - *it2);
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  result_type result_;
};

class JvVisitor3 {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  JvVisitor3(result_type &result) : result_(std::move(result)) {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end) {
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    for (; it1 != value1_end; ++it1, ++it_result) {
      *it_result = *it_result - *it1;
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  result_type result_;
};

class NegativeVisitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  NegativeVisitor() {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end) {
    result_ = result_type(idx_end - idx_begin);
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    for (; it1 != value1_end; ++it1, ++it_result) {
      *it_result = -*it1;
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  result_type result_;
};

class AddVisitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  AddVisitor(result_type &result) : result_(std::move(result)) {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end) {
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    for (; it1 != value1_end; ++it1, ++it_result) {
      *it_result = *it_result + *it1;
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  result_type result_;
};

class Add2Visitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  Add2Visitor(result_type &result) : result_(std::move(result)) {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end, const H &value2_begin,
                  const H &value2_end) {
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    auto it2 = value2_begin;
    for (; it2 != value1_end && it1 != value1_end; ++it1, ++it2, ++it_result) {
      *it_result = *it_result + *it1 + *it2;
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  result_type result_;
};

class Add5Visitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  Add5Visitor(result_type &result) : result_(std::move(result)) {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void
  operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
             const H &value1_end, const H &value2_begin, const H &value2_end,
             const H &value3_begin, const H &value3_end, const H &value4_begin,
             const H &value4_end, const H &value5_begin, const H &value5_end) {
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    auto it2 = value2_begin;
    auto it3 = value3_begin;
    auto it4 = value4_begin;
    auto it5 = value5_begin;
    for (; it1 != value1_end && it2 != value2_end && it3 != value3_end &&
           it4 != value4_end && it5 != value5_end;
         ++it1, ++it2, ++it3, ++it4, ++it5, ++it_result) {
      *it_result = *it_result + *it1 + *it2 + *it3 + *it4 + *it5;
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  result_type result_;
};

class Sum2Visitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  Sum2Visitor() {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end, const H &value2_begin,
                  const H &value2_end) {
    result_ = result_type(idx_end - idx_begin);
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    auto it2 = value2_begin;
    for (; it2 != value1_end && it1 != value1_end; ++it1, ++it2, ++it_result) {
      *it_result = *it1 + *it2;
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  result_type result_;
};

class Sum2DivideVisitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  Sum2DivideVisitor(int divisor) : divisor_(divisor) {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end, const H &value2_begin,
                  const H &value2_end) {
    result_ = result_type(idx_end - idx_begin);
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    auto it2 = value2_begin;
    for (; it2 != value1_end && it1 != value1_end; ++it1, ++it2, ++it_result) {
      *it_result = (*it1 + *it2) / divisor_;
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  int divisor_;
  result_type result_;
};

class Sum4Visitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  Sum4Visitor() {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end, const H &value2_begin,
                  const H &value2_end, const H &value3_begin,
                  const H &value3_end, const H &value4_begin,
                  const H &value4_end) {
    result_ = result_type(idx_end - idx_begin);
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    auto it2 = value2_begin;
    auto it3 = value3_begin;
    auto it4 = value4_begin;
    for (; it1 != value1_end && it2 != value2_end && it3 != value3_end &&
           it4 != value4_end;
         ++it1, ++it2, ++it3, ++it4, ++it_result) {
      *it_result = *it1 + *it2 + *it3 + *it4;
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  result_type result_;
};

class Sum4DivideVisitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  Sum4DivideVisitor(int divisor) : divisor_(divisor) {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end, const H &value2_begin,
                  const H &value2_end, const H &value3_begin,
                  const H &value3_end, const H &value4_begin,
                  const H &value4_end) {
    result_ = result_type(idx_end - idx_begin);
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    auto it2 = value2_begin;
    auto it3 = value3_begin;
    auto it4 = value4_begin;
    for (; it1 != value1_end && it2 != value2_end && it3 != value3_end &&
           it4 != value4_end;
         ++it1, ++it2, ++it3, ++it4, ++it_result) {
      *it_result = (*it1 + *it2 + *it3 + *it4) / divisor_;
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  int divisor_;
  result_type result_;
};

class Sum5Visitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  Sum5Visitor() {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void
  operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
             const H &value1_end, const H &value2_begin, const H &value2_end,
             const H &value3_begin, const H &value3_end, const H &value4_begin,
             const H &value4_end, const H &value5_begin, const H &value5_end) {
    result_ = result_type(idx_end - idx_begin);
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    auto it2 = value2_begin;
    auto it3 = value3_begin;
    auto it4 = value4_begin;
    auto it5 = value5_begin;
    for (; it1 != value1_end && it2 != value2_end && it3 != value3_end &&
           it4 != value4_end && it5 != value5_end;
         ++it1, ++it2, ++it3, ++it4, ++it5, ++it_result) {
      *it_result = *it1 + *it2 + *it3 + *it4 + *it5;
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  result_type result_;
};

class ElabFinalVisitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  ElabFinalVisitor(result_type &result) : result_(std::move(result)) {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end, const H &value2_begin,
                  const H &value2_end, const H &value3_begin,
                  const H &value3_end, const H &value4_begin,
                  const H &value4_end) {
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    auto it2 = value2_begin;
    auto it3 = value3_begin;
    auto it4 = value4_begin;
    for (; it1 != value1_end && it2 != value2_end && it3 != value3_end &&
           it4 != value4_end;
         ++it1, ++it2, ++it3, ++it4, ++it_result) {
      *it_result = *it_result + *it1 - *it2 - *it3 - *it4;
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  result_type result_;
};

class ElabzTFinalVisitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  ElabzTFinalVisitor(result_type &result) : result_(std::move(result)) {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end, const H &value2_begin,
                  const H &value2_end) {
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    auto it2 = value2_begin;
    for (; it1 != value1_end && it2 != value2_end; ++it1, ++it2, ++it_result) {
      *it_result = *it_result - *it1 - *it2;
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  result_type result_;
};

class ExtractDivideVisitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  ExtractDivideVisitor(result_type &result, int divisor)
      : result_(std::move(result)), divisor_(divisor) {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end) {
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    for (; it1 != value1_end; ++it1, ++it_result) {
      *it_result = (*it_result - *it1) / divisor_;
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  int divisor_;
  result_type result_;
};

class Extract2Visitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  Extract2Visitor(result_type &result) : result_(std::move(result)) {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end, const H &value2_begin,
                  const H &value2_end) {
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    auto it2 = value2_begin;
    for (; it1 != value1_end && it2 != value2_end; ++it1, ++it2, ++it_result) {
      *it_result = *it_result - *it1 - *it2;
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  result_type result_;
};

class Extract5Visitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  Extract5Visitor(result_type &result) : result_(std::move(result)) {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void
  operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
             const H &value1_end, const H &value2_begin, const H &value2_end,
             const H &value3_begin, const H &value3_end, const H &value4_begin,
             const H &value4_end, const H &value5_begin, const H &value5_end) {
    auto it_result = result_.begin();
    auto it1 = value1_begin;
    auto it2 = value2_begin;
    auto it3 = value3_begin;
    auto it4 = value4_begin;
    auto it5 = value5_begin;
    for (; it1 != value1_end && it2 != value2_end && it3 != value3_end &&
           it4 != value4_end && it5 != value5_end;
         ++it1, ++it2, ++it3, ++it4, ++it5, ++it_result) {
      *it_result = *it_result - *it1 - *it2 - *it3 - *it4 - *it5;
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  result_type result_;
};

class MeanVisitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef double result_type;

  MeanVisitor() {}
  void pre() {
    result_ = 0;
    count = 0;
  }
  void post() { result_ /= count; }

  template <typename K, typename H>
  void operator()(const K &idx_begin, const H &value) {
    result_ += value;
    count++;
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  int count;
  result_type result_;
};

class MeanSingleVisitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef double result_type;

  MeanSingleVisitor() {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end) {
    int n = idx_end - idx_begin;
    result_ = 0;
    auto it1 = value1_begin;
    // #pragma omp parallel for firstprivate(it1) reduction(+ : result_)
    for (int i = 0; i < n; i++) {
      result_ += it1[i];
    }
    result_ /= n;
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  result_type result_;
};

class GroupbyIndicesVisitor {
public:
  typedef unsigned long index_type;
  typedef int value_type;
  typedef std::vector<int> result_type;

  GroupbyIndicesVisitor(std::unordered_map<int, int> &place) : place_(place) {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end) {
    auto it = value1_begin;
    int n = idx_end - idx_begin;
    result_ = result_type(n);
    for (int i = 0; i < n; i++) {
      result_[i] = place_[it[i]];
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  std::unordered_map<int, int> place_;
  result_type result_;
};

class MeanGroupVisitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  MeanGroupVisitor(std::vector<int> &index, int type_num)
      : index_(std::move(index)) {
    result_ = std::vector<double>(type_num);
  }
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end) {
    std::cout << "size: " << index_.size() << std::endl;
    // for (int i = 0; i < index_.size(); i++) {
    //   std::cout << index_[i] << std::endl;
    // }
    int n = idx_end - idx_begin;
    auto it1 = value1_begin;
    // #pragma omp parallel for firstprivate(it1)
    for (int i = 0; i < n; i++) {
      // std::cout << i << std::endl;
      result_[index_[i]] += it1[i];
    }
    for (int i = 0; i < result_.size(); i++) {
      result_[i] /= n;
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  std::vector<int> index_;
  result_type result_;
};

class GroupbyBordersVisitor {
public:
  typedef unsigned long index_type;
  typedef int value_type;
  typedef std::unordered_map<int, std::tuple<int, int>> result_type;

  GroupbyBordersVisitor(std::unordered_map<int, int> &place) : place_(place) {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end) {
    auto it = value1_begin;
    int n = idx_end - idx_begin;
    for (auto &pl : place_) {
      result_[pl.second] = std::make_tuple(n, 0);
      // std::cout << pl.first << " " << pl.second << std::endl;
    }
    std::tuple<int, int> &tmp = result_[it[0]];
    // std::get<0>(tmp) = 3;
    // std::get<1>(tmp) = 5;
    // for (int i = 0; i < 5; i++) {
    //   std::cout << it[i] << " " << std::get<0>(result_[it[i]]) << " "
    //             << std::get<1>(result_[it[i]]) << std::endl;
    // }
    // for (auto &res : result_) {
    //   std::cout << res.first << " " << std::get<0>(res.second) << " "
    //             << std::get<1>(res.second) << std::endl;
    // }
    int pos1, pos2;
    for (int i = 0; i < n; i++) {
      std::tuple<int, int> &tmp = result_[place_[it[i]]];
      // std::cout << i << " " << std::get<0>(tmp) << " " << std::get<1>(tmp)
      //           << " " << std::get<0>(result_[it[i]]) << " "
      //           << std::get<1>(result_[it[i]]) << " " << it[i] << std::endl;
      if (std::get<0>(tmp) > i) {
        std::get<0>(tmp) = i;
        // std::cout << i << " " << std::get<0>(tmp) << std::endl;
      }
      if (std::get<1>(tmp) < i) {
        std::get<1>(tmp) = i;
      }
      // std::cout << i << " " << std::get<0>(tmp) << " " << std::get<1>(tmp)
      //           << " " << std::get<0>(result_[it[i]]) << " "
      //           << std::get<1>(result_[it[i]]) << " " << it[i] << std::endl;
    }
    // for (auto &res : result_) {
    //   std::cout << res.first << " " << std::get<0>(res.second) << " "
    //             << std::get<1>(res.second) << std::endl;
    // }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  std::unordered_map<int, int> place_;
  result_type result_;
};

class MeanBordersVisitor {
public:
  typedef unsigned long index_type;
  typedef double value_type;
  typedef std::vector<double> result_type;

  MeanBordersVisitor(std::unordered_map<int, std::tuple<int, int>> &borders)
      : borders_(std::move(borders)) {}
  void pre() {}
  void post() {}

  template <typename K, typename H>
  void operator()(const K &idx_begin, const K &idx_end, const H &value1_begin,
                  const H &value1_end) {
    int n = idx_end - idx_begin;
    auto it1 = value1_begin;
    result_ = result_type(borders_.size());
    // for (auto &bord : borders_) {
    //   std::cout << bord.first << " " << std::get<0>(bord.second) << " "
    //             << std::get<1>(bord.second) << std::endl;
    // }
    for (auto &border_pair : borders_) {
      // #pragma omp parallel for firstprivate(border_pair, it1)
      for (int i = std::get<0>(border_pair.second);
           i <= std::get<1>(border_pair.second); i++) {
        // std::cout << i << std::endl;
        result_[border_pair.first] += it1[i];
      }
      result_[border_pair.first] /= (std::get<1>(border_pair.second) -
                                     std::get<0>(border_pair.second) + 1);
    }
  }

  result_type &get_result() { return result_; }
  const result_type &get_result() const { return result_; }

private:
  std::unordered_map<int, std::tuple<int, int>> borders_;
  result_type result_;
};