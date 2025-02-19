#include "data_processing.h"
#include "jackknife.h"
#include "visitors.h"

#include "omp.h"
#include <DataFrame/DataFrame.h>
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

std::vector<int> get_radii_sq(int box_size) {
  int x, y;
  std::set<int> radii;
  for (int i = 0; i < 2; i++) {
    y = box_size;
    x = box_size - i;
    while (x * x + y * y > (box_size - 1) * (box_size - 1)) {
      radii.insert(x * x + y * y);
      x -= 1;
      y -= 1;
    }
  }
  radii.insert(box_size * box_size);
  std::vector<int> result(radii.begin(), radii.end());
  std::reverse(result.begin(), result.end());
  return result;
}

std::vector<hmdf::StdDataFrame<unsigned long>>
split_df(hmdf::StdDataFrame<unsigned long> &df) {
  std::vector<hmdf::StdDataFrame<unsigned long>> dfs;
  hmdf::StdDataFrame<unsigned long> df_tmp;
  std::vector<int> unique_conf = df.get_col_unique_values<int>("conf_end");
  for (auto conf : unique_conf) {
    auto functor = [conf](const unsigned long &, const int &conf_end) -> bool {
      return conf_end == conf;
    };
    df_tmp = df.get_data_by_sel<int, decltype(functor), int, double>("conf_end",
                                                                     functor);
    dfs.push_back(df_tmp);
  }
  return dfs;
}

void df_cut_box(std::vector<hmdf::StdDataFrame<unsigned long>> &dfs,
                int coord_max, int cut) {
  auto functor_cut = [coord_max, cut](const unsigned long &, const int &x,
                                      const int &y) -> bool {
    return (x <= 2 * coord_max - cut) && (x >= cut) &&
           (y <= 2 * coord_max - cut) && (y >= cut);
  };
  for (int i = 0; i < dfs.size(); i++) {
    dfs[i] =
        dfs[i].get_data_by_sel<int, int, decltype(functor_cut), int, double>(
            "x", "y", functor_cut);
  }
}

void df_cut_rad(std::vector<hmdf::StdDataFrame<unsigned long>> &dfs,
                int coord_max, int rad_cut) {
  hmdf::StdDataFrame<unsigned long> df_tmp;
  auto functor_rad_cut = [coord_max, rad_cut](const unsigned long &,
                                              const int &x,
                                              const int &y) -> bool {
    return (x - coord_max) * (x - coord_max) +
               (y - coord_max) * (y - coord_max) <=
           rad_cut;
  };
  // #pragma omp parallel for firstprivate(functor_rad_cut) private(df_tmp)
  for (int i = 0; i < dfs.size(); i++) {
    df_tmp =
        dfs[i]
            .get_data_by_sel<int, int, decltype(functor_rad_cut), int, double>(
                "x", "y", functor_rad_cut);
    dfs[i] = df_tmp;
  }
}

std::vector<std::vector<double>>
observables_aver(std::vector<hmdf::StdDataFrame<unsigned long>> &dfs) {
  std::vector<std::vector<double>> result(16);
  for (int i = 0; i < 16; i++) {
    result.reserve(dfs.size());
  }
  std::vector<std::string> observables = {
      "S",   "Jv", "Jv1", "Jv2",    "Blab",    "E",  "Elab", "Bz",
      "Bxy", "Ez", "Exy", "ElabzT", "ElabxyT", "Ae", "Am",   "AlabeT"};
  for (auto &df_tmp : dfs) {
    for (int i = 0; i < observables.size(); i++) {
      MeanSingleVisitor mean_single_visitor;
      df_tmp.single_act_visit<double>(observables[i].c_str(),
                                      mean_single_visitor);
      result[i].push_back(mean_single_visitor.get_result());
    }
  }
  return result;
}

std::vector<std::vector<double>>
observables_aver(hmdf::StdDataFrame<unsigned long> &df) {
  std::vector<std::vector<double>> result(16);
  std::vector<std::string> observables = {
      "S",   "Jv", "Jv1", "Jv2",    "Blab",    "E",  "Elab", "Bz",
      "Bxy", "Ez", "Exy", "ElabzT", "ElabxyT", "Ae", "Am",   "AlabeT"};

  std::unordered_map<int, int> place_map = make_map_place(df);
#pragma omp parallel for firstprivate(place_map)
  for (int i = 0; i < observables.size(); i++) {
    GroupbyBordersVisitor groupby_borders_visitor(place_map);
    df.single_act_visit<int>("conf_end", groupby_borders_visitor);
    MeanBordersVisitor mean_borders_visitor(
        groupby_borders_visitor.get_result());
    df.single_act_visit<double>(observables[i].c_str(), mean_borders_visitor);
    result[i] = mean_borders_visitor.get_result();
  }
  return result;
}

std::map<std::string, std::tuple<double, double>>
jackknife(std::vector<std::vector<double>> &data, int bin_size) {
  std::vector<std::string> observables = {
      "S",   "Jv", "Jv1", "Jv2",    "Blab",    "E",  "Elab", "Bz",
      "Bxy", "Ez", "Exy", "ElabzT", "ElabxyT", "Ae", "Am",   "AlabeT"};
  std::vector<unsigned long> bin_borders =
      get_bin_borders(data[0].size(), bin_size);
  std::vector<std::vector<double>> jackknife_vec =
      do_jackknife(data, bin_borders);
  std::map<std::string, std::tuple<double, double>> result;
  for (int i = 0; i < jackknife_vec.size(); i++) {
    result[observables[i]] = get_aver(jackknife_vec[i]);
  }
  return result;
}

std::unordered_map<int, int>
make_map_place(hmdf::StdDataFrame<unsigned long> &df) {
  std::unordered_map<int, int> place;
  std::vector<int> unique_conf = df.get_col_unique_values<int>("conf_end");
  for (int i = 0; i < unique_conf.size(); i++) {
    place[unique_conf[i]] = i;
  }
  return place;
}
