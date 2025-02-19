#pragma once

#include <algorithm>
#include <map>
#include <vector>

#pragma omp declare reduction(                                                 \
        vec_double_plus : std::vector<double> : std::transform(                \
                omp_out.begin(), omp_out.end(), omp_in.begin(),                \
                    omp_out.begin(), std::plus<double>()))                     \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

std::vector<unsigned long> get_bin_borders(unsigned long data_size,
                                           unsigned long bin_size);
std::tuple<double, double> get_aver(std::vector<double> &jackknife_vec);

template <typename H>
std::vector<double> do_jackknife(const H &data_begin, const H &data_end,
                                 std::vector<unsigned long> &bin_borders) {
  unsigned long n = data_end - data_begin;
  double sum = 0;
#pragma omp parallel for reduction(+ : sum) firstprivate(data_begin, n)
  for (unsigned long i = 0; i < n; i++) {
    sum += data_begin[i];
  }

  unsigned long l = bin_borders.size() - 1;
  std::vector<double> data_jackknife(l);
#pragma omp parallel for shared(data_jackknife, sum, bin_borders)              \
    firstprivate(data_begin)
  for (unsigned long i = 0; i < l; i++) {
    data_jackknife[i] = sum;
    for (unsigned long k = bin_borders[i]; k < bin_borders[i + 1]; k++) {
      data_jackknife[i] -= data_begin[k];
    }
    data_jackknife[i] =
        data_jackknife[i] / (n - bin_borders[i + 1] + bin_borders[i]);
  }
  return data_jackknife;
}

std::vector<std::vector<double>>
do_jackknife(std::vector<std::vector<double>> &data,
             std::vector<unsigned long> &bin_borders);