#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "read_data.h"

std::vector<std::string> parse_line(std::string &line) {
  std::vector<std::string> result;
  std::string delimiter = ",";
  size_t pos = 0;
  std::string token;
  while ((pos = line.find(delimiter)) != std::string::npos) {
    token = line.substr(0, pos);
    result.push_back(token);
    line.erase(0, pos + delimiter.length());
  }
  result.push_back(line);
  return result;
}

void read_csv(
    std::string file_path,
    std::map<std::tuple<int, int, int, int>, std::vector<double>> &data) {
  std::ifstream file_stream(file_path);
  std::string line;
  std::vector<std::string> parsed_line;
  std::getline(file_stream, line);
  // std::vector<int> smearing_points = {11, 21, 31, 41};
  while (std::getline(file_stream, line)) {
    parsed_line = parse_line(line);
    // if (std::stoi(parsed_line[0]) <= smearing_max &&
    //     std::stoi(parsed_line[0]) >= 1 &&
    //     std::stoi(parsed_line[1]) <= smearing_max &&
    // if (std::find(smearing_points.begin(), smearing_points.end(),
    //               std::stoi(parsed_line[0])) != smearing_points.end() &&
    //     std::find(smearing_points.begin(), smearing_points.end(),
    //               std::stoi(parsed_line[1])) != smearing_points.end() &&
    //     std::stoi(parsed_line[2]) >= 1) {
    data[{std::stoi(parsed_line[3]), std::stoi(parsed_line[2]),
          std::stoi(parsed_line[0]), std::stoi(parsed_line[1])}]
        .push_back(double(std::stod(parsed_line[4])));
    // }
  }
}

void read_csv_functional(
    std::string file_path,
    std::map<std::tuple<std::string, int, int>, double> &functional,
    std::string chain, int conf_num) {
  std::ifstream file_stream(file_path);
  std::string line;
  std::vector<std::string> parsed_line;
  std::getline(file_stream, line);
  while (std::getline(file_stream, line)) {
    parsed_line = parse_line(line);
    functional[{chain, conf_num, std::stoi(parsed_line[0])}] =
        std::stod(parsed_line[1]);
  }
}

std::map<std::tuple<int, int>, std::vector<std::vector<double>>>
read_data(std::string dir_path, std::string file_start, std::string file_end,
          int padding, int num_max) {
  std::map<std::tuple<int, int, int, int>, std::vector<double>> data;
  std::string file_path;
  std::vector<std::string> chains = {"",    "s0/", "s1/", "s2/", "s3/", "s4/",
                                     "s5/", "s6/", "s7/", "s8/", "s9/", "s10/"};
  // std::vector<std::string> chains = {"s1/"};
  for (auto chain : chains) {
    for (int i = 1; i <= num_max; i++) {
      std::stringstream ss;
      ss << dir_path << "/" << chain << file_start << std::setw(padding)
         << std::setfill('0') << std::to_string(i) << file_end;
      file_path = ss.str();
      // std::cout << file_path << std::endl;
      if (std::filesystem::exists(file_path))
        read_csv(file_path, data);
    }
  }
  std::map<std::tuple<int, int>, std::vector<std::vector<double>>> result;
  for (const auto &pair : data) {
    result[{std::get<0>(pair.first), std::get<1>(pair.first)}].push_back(
        std::move(data[pair.first]));
  }
  return result;
}

std::map<std::tuple<int, int>, std::vector<std::vector<double>>>
read_data_copies(std::string dir_path, std::string file_start,
                 std::string file_end, int padding, int num_max, int copy) {
  std::map<std::tuple<int, int, int, int>, std::vector<double>> data;
  std::string file_path;
  std::vector<std::string> chains = {"",    "s0/", "s1/", "s2/", "s3/", "s4/",
                                     "s5/", "s6/", "s7/", "s8/", "s9/", "s10/"};
  for (auto chain : chains) {
    for (int i = 0; i <= num_max; i++) {
      for (int j = copy; j >= 0; j--) {
        std::stringstream ss;
        ss << dir_path << "/" << chain << file_start << std::setw(padding)
           << std::setfill('0') << std::to_string(i) << "_" << std::to_string(j)
           << file_end;
        file_path = ss.str();
        // std::cout << file_path << std::endl;
        if (std::filesystem::exists(file_path)) {
          read_csv(file_path, data);
          break;
        }
      }
    }
  }
  std::map<std::tuple<int, int>, std::vector<std::vector<double>>> result;
  for (const auto &pair : data) {
    result[{std::get<0>(pair.first), std::get<1>(pair.first)}].push_back(
        std::move(data[pair.first]));
  }
  return result;
}

std::map<std::tuple<std::string, int, int>, double>
read_functional(std::string dir_path, int padding, int num_max) {
  std::map<std::tuple<std::string, int, int>, double> functional;
  std::string file_path;
  std::vector<std::string> chains = {"/",   "s0/", "s1/", "s2/", "s3/", "s4/",
                                     "s5/", "s6/", "s7/", "s8/", "s9/", "s10/"};
  for (auto chain : chains) {
    for (int i = 1; i <= num_max; i++) {
      std::stringstream ss;
      ss << dir_path << "/" << chain << "functional_" << std::setw(padding)
         << std::setfill('0') << std::to_string(i);
      file_path = ss.str();
      if (std::filesystem::exists(file_path)) {
        read_csv_functional(file_path, functional, chain, i);
      }
    }
  }
  return functional;
}

std::map<std::tuple<int, int>, std::vector<std::vector<double>>>
read_data_bins(std::string dir_path, std::string file_start,
               std::string file_end, int padding,
               std::map<std::tuple<std::string, int, int>, double> &functional,
               double bin_left, double bin_right, double &functional_average) {
  std::map<std::tuple<int, int, int, int>, std::vector<double>> data;
  std::string file_path;
  std::vector<std::string> chains = {"/",   "s0/", "s1/", "s2/", "s3/", "s4/",
                                     "s5/", "s6/", "s7/", "s8/", "s9/", "s10/"};
  functional_average = 0;
  int count = 0;
  for (const auto &pair : functional) {
    if (pair.second >= bin_left && pair.second < bin_right) {
      std::stringstream ss;
      ss << dir_path << "/" << std::get<0>(pair.first) << file_start
         << std::setw(padding) << std::setfill('0')
         << std::to_string(std::get<1>(pair.first)) << "_"
         << std::to_string(std::get<2>(pair.first)) << file_end;
      file_path = ss.str();
      // std::cout << file_path << std::endl;
      if (std::filesystem::exists(file_path)) {
        functional_average += pair.second;
        count += 1;
        read_csv(file_path, data);
      }
    }
  }
  functional_average /= count;
  std::map<std::tuple<int, int>, std::vector<std::vector<double>>> result;
  for (const auto &pair : data) {
    result[{std::get<0>(pair.first), std::get<1>(pair.first)}].push_back(
        std::move(data[pair.first]));
  }
  return result;
}