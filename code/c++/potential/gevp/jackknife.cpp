#include "jackknife.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>

std::vector<unsigned long>
accumulate_bins(std::vector<unsigned long> &bin_sizes) {
  unsigned long bin_num = bin_sizes.size();
  std::vector<unsigned long> bin_borders;
  bin_borders.reserve(bin_num + 1);
  bin_borders.push_back(0);
  unsigned long tmp = 0;
  for (unsigned long i = 0; i < bin_num; i++) {
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

#pragma omp declare reduction(                                                 \
        vec_double_plus : std::vector<double> : std::transform(                \
                omp_out.begin(), omp_out.end(), omp_in.begin(),                \
                    omp_out.begin(), std::plus<double>()))                     \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

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

std::vector<double>
make_gevp(const std::vector<std::vector<double>> &data_jackknife_0,
          const std::vector<std::vector<double>> &data_jackknife_t,
          bool test_bool) {
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;
  int matrix_size = (round(sqrt(1 + 8 * data_jackknife_0.size())) - 1) / 2;
  Eigen::MatrixXd A(matrix_size, matrix_size);
  Eigen::MatrixXd B(matrix_size, matrix_size);
  std::vector<double> lambdas(data_jackknife_0[0].size());
  // #pragma omp parallel for firstprivate(A, B, ges)                               \
    shared(data_jackknife_0, data_jackknife_t, lambdas)
  for (int i = 0; i < data_jackknife_0[0].size(); i++) {
    int count = 0;
    for (int j = 0; j < matrix_size; j++) {
      for (int k = j; k < matrix_size; k++) {
        // if (j == k) {
        //   B(j, k) = 1;
        // } else {
        //   B(j, k) = 0;
        //   B(k, j) = 0;
        // }
        A(j, k) = data_jackknife_t[count][i];
        B(j, k) = data_jackknife_0[count][i];
        A(k, j) = data_jackknife_t[count][i];
        B(k, j) = data_jackknife_0[count][i];
        count++;
      }
    }
    if (test_bool) {
      std::cout << A << std::endl << std::endl;
    }
    ges.compute(A, B, Eigen::DecompositionOptions::EigenvaluesOnly);
    auto eigenvalues = ges.eigenvalues();
    if (test_bool) {
      for (int i = 0; i < eigenvalues.size(); i++) {
        std::cout << eigenvalues[i] << " ";
      }
      std::cout << std::endl << std::endl;
    }
    lambdas[i] = eigenvalues[eigenvalues.size() - 1];
  }
  return lambdas;
}

std::map<int, std::vector<int>> get_sizes(
    std::map<std::tuple<int, int>, std::vector<std::vector<double>>> &data) {
  std::map<int, std::vector<int>> sizes;
  for (const auto &pair : data) {
    sizes[std::get<0>(pair.first)].push_back(std::get<1>(pair.first));
  }
  for (const auto &pair : sizes) {
    std::sort(sizes[pair.first].begin(), sizes[pair.first].end());
  }
  return sizes;
}

std::tuple<double, double> potential_aver(std::vector<double> &lambda_t1,
                                          std::vector<double> &lambda_t2,
                                          bool test_bool) {
  long int n = lambda_t1.size();
  std::vector<double> tmp(n);
  double a;
  // #pragma omp parallel for shared(lambda_t1, lambda_t2, tmp) private(a)
  for (long int i = 0; i < n; i++) {
    a = lambda_t1[i] / lambda_t2[i];
    if (test_bool) {
      std::cout << lambda_t1[i] << " " << lambda_t2[i] << " " << std::log(a)
                << std::endl;
    }
    if (a > 0)
      tmp[i] = std::log(a);
    else
      tmp[i] = 0;
  }
  double aver = 0;
  double sigma = 0;
#pragma omp parallel for shared(tmp) reduction(+ : aver)
  for (long int i = 0; i < n; i++) {
    aver += tmp[i];
  }
  aver /= n;
#pragma omp parallel for shared(tmp) firstprivate(aver) reduction(+ : sigma)
  for (long int i = 0; i < n; i++) {
    sigma += (tmp[i] - aver) * (tmp[i] - aver);
  }
  return {aver, sqrt((n - 1) / (n + .0) * sigma)};
}

std::map<std::tuple<int, int>, std::tuple<double, double>> calculate_potential(
    std::map<std::tuple<int, int>, std::vector<std::vector<double>>> &data,
    int bin_size, int t0) {
  std::map<std::tuple<int, int>, std::tuple<double, double>> potential;
  std::map<int, std::vector<int>> sizes = get_sizes(data);
  std::vector<unsigned long> bin_borders =
      get_bin_borders(data[data.begin()->first][0].size(), bin_size);
  bool test_bool = false;
  for (const auto &pair : sizes) {
    for (int i = 0; i < pair.second.size(); i++) {
      std::cout << pair.first << " " << pair.second[i] << std::endl;
    }
    std::vector<std::vector<double>> lambdas;
    std::vector<std::vector<double>> data_jackknife_0 =
        do_jackknife(data[{pair.first, pair.second[t0]}], bin_borders);
    for (int t = t0 + 1; t < pair.second.size(); t++) {
      std::vector<std::vector<double>> data_jackknife_t =
          do_jackknife(data[{pair.first, pair.second[t]}], bin_borders);
      if (pair.first == 20 && t == 18) {
        test_bool = true;
      } else {
        test_bool = false;
      }
      lambdas.push_back(
          make_gevp(data_jackknife_0, data_jackknife_t, test_bool));
      // if (pair.first == 20 && t == 19) {
      //   for (int i = 0; i < lambdas[t - 1].size(); i++) {
      //     std::cout << lambdas[t - 1][i] << std::endl;
      //   }
      // }
    }
    for (int t = t0 + 1; t < pair.second.size() - 1; t++) {
      if (pair.first == 20 && t == 8) {
        test_bool = true;
      } else {
        test_bool = false;
      }
      potential[{pair.first, pair.second[t]}] =
          potential_aver(lambdas[t - 1 - t0], lambdas[t - t0], test_bool);
    }
  }
  return potential;
}