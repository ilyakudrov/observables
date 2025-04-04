#pragma once

#include <map>
#include <unordered_map>
#include <vector>

#include <DataFrame/DataFrame.h>

std::vector<hmdf::StdDataFrame<unsigned long>>
split_df(hmdf::StdDataFrame<unsigned long> &df);
std::vector<int> get_radii_sq(int square_size);
void df_cut_box(std::vector<hmdf::StdDataFrame<unsigned long>> &dfs,
                int coord_max, int cut);
void df_cut_rad(std::vector<hmdf::StdDataFrame<unsigned long>> &dfs,
                int coord_max, int rad_cut);
std::vector<std::vector<double>>
observables_aver(std::vector<hmdf::StdDataFrame<unsigned long>> &dfs);
std::vector<std::vector<double>>
observables_aver(hmdf::StdDataFrame<unsigned long> &df);
std::vector<std::vector<double>>
observables_aver_polyakov(hmdf::StdDataFrame<unsigned long> &df);
std::map<std::string, std::tuple<double, double>>
jackknife(std::vector<std::vector<double>> &data, int bin_size);
std::map<std::string, std::tuple<double, double>>
jackknife_polyakov(std::vector<std::vector<double>> &data, int bin_size);
std::unordered_map<int, int>
make_map_place(hmdf::StdDataFrame<unsigned long> &df);