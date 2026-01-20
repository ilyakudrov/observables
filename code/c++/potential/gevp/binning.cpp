#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "binning.h"

std::vector<double>
get_bin_edges(std::map<std::tuple<std::string, int, int>, double> &functional) {
  std::vector<double> values;
  for (const auto &pair : functional) {
    values.push_back(pair.second);
  }
  std::sort(values.begin(), values.end());
  double min = values[0];
  double max = values[values.size() - 1];
  double iqr = values[values.size() * 3 / 4] - values[values.size() / 4];
  double bin_width = 2 * iqr / std::cbrt(values.size());
  std::vector<double> bin_edges;
  double right_edge = values[0] * 0.9999999;
  bin_edges.push_back(right_edge);
  while (right_edge <= max) {
    right_edge += bin_width;
    bin_edges.push_back(right_edge);
  }
  return bin_edges;
}

std::map<double, std::vector<std::tuple<std::string, int, int>>> bin_functional(
    std::map<std::tuple<std::string, int, int>, double> &functional) {
  std::map<double, std::vector<std::tuple<std::string, int, int>>> bins;
  std::vector<double> bin_edges = get_bin_edges(functional);
  double bin_value;
  for (int i = 1; i < bin_edges.size(); i++) {
    bin_value = (bin_edges[i] + bin_edges[i - 1]) / 2;
    for (const auto &pair : functional) {
      if (pair.second >= bin_edges[i - 1] && pair.second < bin_edges[i]) {
        bins[bin_value].push_back(pair.first);
      }
    }
  }
  return bins;
}