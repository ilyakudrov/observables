#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

std::map<std::tuple<double, double, double, double, int, int, int, int>,
         std::tuple<std::vector<double>, std::vector<double>>>
read_data(std::string dir_path, std::string file_start, std::string file_end,
          int padding, int num_max);
std::set<std::tuple<double, double, double, double>>
read_momenta(std::string dir_path, std::string file_start, std::string file_end,
             int padding, int num_max);
std::map<std::tuple<int, int, int, int>,
         std::tuple<std::vector<double>, std::vector<double>>>
read_data_single_momentum(std::tuple<double, double, double, double> &momentum,
                          std::string dir_path, std::string file_start,
                          std::string file_end, int padding, int num_max);
std::map<std::tuple<double, double, double, double>,
         std::tuple<std::vector<double>, std::vector<double>>>
read_data_single_component(std::tuple<int, int, int, int> &component,
                           std::string dir_path, std::string file_start,
                           std::string file_end, int padding, int num_max);
std::map<std::tuple<double, double, double, double, int, int>,
         std::tuple<std::vector<double>, std::vector<double>>>
read_data_color_components(std::tuple<int, int> &colors, std::string dir_path,
                           std::string file_start, std::string file_end,
                           int padding, int num_max);
std::map<std::tuple<double, double, double, double, int>,
         std::tuple<std::vector<double>, std::vector<double>>>
read_data_diagonal(std::string dir_path, std::string file_start,
                   std::string file_end, int padding, int num_max);