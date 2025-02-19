#include "jackknife.h"

#include <cmath>
#include <vector>

std::vector<unsigned long>
accumulate_bins(std::vector<unsigned long> &bin_sizes) {
  unsigned long bin_num = bin_sizes.size();
  std::vector<unsigned long> bin_borders;
  bin_borders.reserve(bin_num + 1);
  bin_borders.push_back(0);
  unsigned long tmp = 0;
  for (long int i = 0; i < bin_num; i++) {
    tmp += bin_sizes[i];
    bin_borders.push_back(tmp);
  }
  return bin_borders;
}

std::vector<unsigned long> get_bin_borders(unsigned long data_size,
                                           unsigned long bin_size) {
  unsigned long nbins = data_size / bin_size;
  std::vector<unsigned long> bin_sizes(nbins, bin_size);
  unsigned long residual_size = data_size - nbins * bin_size;
  unsigned long idx = 0;
  while (residual_size > 0) {
    bin_sizes[idx] += 1;
    residual_size -= 1;
    idx = (idx + 1) % nbins;
  }
  return accumulate_bins(bin_sizes);
}

std::tuple<double, double> get_aver(std::vector<double> &data) {
  int n = data.size();
  double aver = 0;
  double sigma = 0;
#pragma omp parallel for reduction(+ : aver)
  for (long int i = 0; i < n; i++) {
    aver += data[i];
  }
  aver /= n;
#pragma omp parallel for firstprivate(aver) reduction(+ : sigma)
  for (long int i = 0; i < n; i++) {
    sigma += (data[i] - aver) * (data[i] - aver);
  }
  return {aver, sqrt((n - 1) / (n + .0) * sigma)};
}

std::vector<std::vector<double>>
do_jackknife(std::vector<std::vector<double>> &data,
             std::vector<unsigned long> &bin_borders) {
  int m = data.size();
  unsigned long n = data[0].size();
  std::vector<double> sum(m);
#pragma omp parallel for collapse(2) reduction(vec_double_plus : sum)          \
    shared(data)
  for (int i = 0; i < m; i++) {
    for (unsigned long j = 0; j < n; j++) {
      sum[i] += data[i][j];
    }
  }

  std::vector<std::vector<double>> data_jackknife(
      m, std::vector<double>(bin_borders.size() - 1));
  long int l = bin_borders.size() - 1;
#pragma omp parallel for collapse(2) shared(data_jackknife, sum, bin_borders)
  for (int i = 0; i < m; i++) {
    for (long int j = 0; j < l; j++) {
      data_jackknife[i][j] = sum[i];
      for (long int k = bin_borders[j]; k < bin_borders[j + 1]; k++) {
        data_jackknife[i][j] -= data[i][k];
      }
      data_jackknife[i][j] =
          data_jackknife[i][j] / (n - bin_borders[j + 1] + bin_borders[j]);
    }
  }

  return data_jackknife;
}