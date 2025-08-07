#include "read_data.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

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
    std::map<std::tuple<double, double, double, double, int, int, int, int>,
             std::tuple<std::vector<double>, std::vector<double>>> &data) {
  std::ifstream file_stream(file_path);
  std::string line;
  std::vector<std::string> parsed_line;
  std::getline(file_stream, line);
  while (std::getline(file_stream, line)) {
    parsed_line = parse_line(line);
    std::get<0>(data[{std::stod(parsed_line[0]), std::stod(parsed_line[1]),
                      std::stod(parsed_line[2]), std::stod(parsed_line[3]),
                      std::stoi(parsed_line[4]), std::stoi(parsed_line[5]),
                      std::stoi(parsed_line[6]), std::stoi(parsed_line[7])}])
        .push_back(double(std::stod(parsed_line[8])));
    std::get<1>(data[{std::stod(parsed_line[0]), std::stod(parsed_line[1]),
                      std::stod(parsed_line[2]), std::stod(parsed_line[3]),
                      std::stoi(parsed_line[4]), std::stoi(parsed_line[5]),
                      std::stoi(parsed_line[6]), std::stoi(parsed_line[7])}])
        .push_back(double(std::stod(parsed_line[9])));
  }
}

std::map<std::tuple<double, double, double, double, int, int, int, int>,
         std::tuple<std::vector<double>, std::vector<double>>>
read_data(std::string dir_path, std::string file_start, std::string file_end,
          int padding, int num_max) {
  std::map<std::tuple<double, double, double, double, int, int, int, int>,
           std::tuple<std::vector<double>, std::vector<double>>>
      data;
  std::string file_path;
  std::vector<std::string> chains = {"",    "s0/", "s1/", "s2/", "s3/", "s4/",
                                     "s5/", "s6/", "s7/", "s8/", "s9/", "s10/"};
  for (auto chain : chains) {
    for (int i = 1; i <= num_max; i++) {
      std::stringstream ss;
      ss << dir_path << "/" << chain << file_start << std::setw(padding)
         << std::setfill('0') << std::to_string(i) << file_end;
      file_path = ss.str();
      if (std::filesystem::exists(file_path))
        read_csv(file_path, data);
    }
  }
  return data;
}

void read_momenta_from_csv(
    std::string file_path,
    std::set<std::tuple<double, double, double, double>> &momenta) {
  std::ifstream file_stream(file_path);
  std::string line;
  std::vector<std::string> parsed_line;
  std::getline(file_stream, line);
  while (std::getline(file_stream, line)) {
    parsed_line = parse_line(line);
    momenta.insert({std::stod(parsed_line[0]), std::stod(parsed_line[1]),
                    std::stod(parsed_line[2]), std::stod(parsed_line[3])});
  }
}

std::set<std::tuple<double, double, double, double>>
read_momenta(std::string dir_path, std::string file_start, std::string file_end,
             int padding, int num_max) {
  std::set<std::tuple<double, double, double, double>> momenta;
  std::string file_path;
  std::vector<std::string> chains = {"",    "s0/", "s1/", "s2/", "s3/", "s4/",
                                     "s5/", "s6/", "s7/", "s8/", "s9/", "s10/"};
  for (auto chain : chains) {
    for (int i = 1; i <= num_max; i++) {
      std::stringstream ss;
      ss << dir_path << "/" << chain << file_start << std::setw(padding)
         << std::setfill('0') << std::to_string(i) << file_end;
      file_path = ss.str();
      if (std::filesystem::exists(file_path))
        read_momenta_from_csv(file_path, momenta);
    }
  }
  return momenta;
}

void read_csv_single_momentum(
    std::tuple<double, double, double, double> &momentum, std::string file_path,
    std::map<std::tuple<int, int, int, int>,
             std::tuple<std::vector<double>, std::vector<double>>> &data) {
  std::ifstream file_stream(file_path);
  std::string line;
  std::vector<std::string> parsed_line;
  std::getline(file_stream, line);
  while (std::getline(file_stream, line)) {
    parsed_line = parse_line(line);
    if (std::stod(parsed_line[0]) == std::get<0>(momentum) &&
        std::stod(parsed_line[1]) == std::get<1>(momentum) &&
        std::stod(parsed_line[2]) == std::get<2>(momentum) &&
        std::stod(parsed_line[3]) == std::get<3>(momentum)) {
      std::get<0>(data[{std::stoi(parsed_line[4]), std::stoi(parsed_line[5]),
                        std::stoi(parsed_line[6]), std::stoi(parsed_line[7])}])
          .push_back(double(std::stod(parsed_line[8])));
      std::get<1>(data[{std::stoi(parsed_line[4]), std::stoi(parsed_line[5]),
                        std::stoi(parsed_line[6]), std::stoi(parsed_line[7])}])
          .push_back(double(std::stod(parsed_line[9])));
    }
  }
}

std::map<std::tuple<int, int, int, int>,
         std::tuple<std::vector<double>, std::vector<double>>>
read_data_single_momentum(std::tuple<double, double, double, double> &momentum,
                          std::string dir_path, std::string file_start,
                          std::string file_end, int padding, int num_max) {
  std::map<std::tuple<int, int, int, int>,
           std::tuple<std::vector<double>, std::vector<double>>>
      data;
  std::string file_path;
  std::vector<std::string> chains = {"",    "s0/", "s1/", "s2/", "s3/", "s4/",
                                     "s5/", "s6/", "s7/", "s8/", "s9/", "s10/"};
  for (auto chain : chains) {
    for (int i = 1; i <= num_max; i++) {
      std::stringstream ss;
      ss << dir_path << "/" << chain << file_start << std::setw(padding)
         << std::setfill('0') << std::to_string(i) << file_end;
      file_path = ss.str();
      if (std::filesystem::exists(file_path))
        read_csv_single_momentum(momentum, file_path, data);
    }
  }
  return data;
}

void read_csv_single_component(
    std::tuple<int, int, int, int> &component, std::string file_path,
    std::map<std::tuple<double, double, double, double>,
             std::tuple<std::vector<double>, std::vector<double>>> &data) {
  std::ifstream file_stream(file_path);
  std::string line;
  std::vector<std::string> parsed_line;
  std::getline(file_stream, line);
  while (std::getline(file_stream, line)) {
    parsed_line = parse_line(line);
    if (std::stoi(parsed_line[4]) == std::get<0>(component) &&
        std::stoi(parsed_line[5]) == std::get<1>(component) &&
        std::stoi(parsed_line[6]) == std::get<2>(component) &&
        std::stoi(parsed_line[7]) == std::get<3>(component)) {
      std::get<0>(data[{std::stod(parsed_line[0]), std::stod(parsed_line[1]),
                        std::stod(parsed_line[2]), std::stod(parsed_line[3])}])
          .push_back(double(std::stod(parsed_line[8])));
      std::get<1>(data[{std::stod(parsed_line[0]), std::stod(parsed_line[1]),
                        std::stod(parsed_line[2]), std::stod(parsed_line[3])}])
          .push_back(double(std::stod(parsed_line[9])));
    }
  }
}

std::map<std::tuple<double, double, double, double>,
         std::tuple<std::vector<double>, std::vector<double>>>
read_data_single_component(std::tuple<int, int, int, int> &component,
                           std::string dir_path, std::string file_start,
                           std::string file_end, int padding, int num_max) {
  std::map<std::tuple<double, double, double, double>,
           std::tuple<std::vector<double>, std::vector<double>>>
      data;
  std::string file_path;
  std::vector<std::string> chains = {"",    "s0/", "s1/", "s2/", "s3/", "s4/",
                                     "s5/", "s6/", "s7/", "s8/", "s9/", "s10/"};
  for (auto chain : chains) {
    for (int i = 1; i <= num_max; i++) {
      std::stringstream ss;
      ss << dir_path << "/" << chain << file_start << std::setw(padding)
         << std::setfill('0') << std::to_string(i) << file_end;
      file_path = ss.str();
      if (std::filesystem::exists(file_path))
        read_csv_single_component(component, file_path, data);
    }
  }
  return data;
}

void read_csv_color_components(
    std::tuple<int, int> &colors, std::string file_path,
    std::map<std::tuple<double, double, double, double, int, int>,
             std::tuple<std::vector<double>, std::vector<double>>> &data) {
  std::ifstream file_stream(file_path);
  std::string line;
  std::vector<std::string> parsed_line;
  std::getline(file_stream, line);
  while (std::getline(file_stream, line)) {
    parsed_line = parse_line(line);
    if (std::stoi(parsed_line[6]) == std::get<0>(colors) &&
        std::stoi(parsed_line[7]) == std::get<1>(colors)) {
      std::get<0>(data[{std::stod(parsed_line[0]), std::stod(parsed_line[1]),
                        std::stod(parsed_line[2]), std::stod(parsed_line[3]),
                        std::stoi(parsed_line[4]), std::stoi(parsed_line[5])}])
          .push_back(double(std::stod(parsed_line[8])));
      std::get<1>(data[{std::stod(parsed_line[0]), std::stod(parsed_line[1]),
                        std::stod(parsed_line[2]), std::stod(parsed_line[3]),
                        std::stoi(parsed_line[4]), std::stoi(parsed_line[5])}])
          .push_back(double(std::stod(parsed_line[9])));
    }
  }
}

std::map<std::tuple<double, double, double, double, int, int>,
         std::tuple<std::vector<double>, std::vector<double>>>
read_data_color_components(std::tuple<int, int> &colors, std::string dir_path,
                           std::string file_start, std::string file_end,
                           int padding, int num_max) {
  std::map<std::tuple<double, double, double, double, int, int>,
           std::tuple<std::vector<double>, std::vector<double>>>
      data;
  std::string file_path;
  std::vector<std::string> chains = {"",    "s0/", "s1/", "s2/", "s3/", "s4/",
                                     "s5/", "s6/", "s7/", "s8/", "s9/", "s10/"};
  for (auto chain : chains) {
    for (int i = 1; i <= num_max; i++) {
      std::stringstream ss;
      ss << dir_path << "/" << chain << file_start << std::setw(padding)
         << std::setfill('0') << std::to_string(i) << file_end;
      file_path = ss.str();
      if (std::filesystem::exists(file_path))
        read_csv_color_components(colors, file_path, data);
    }
  }
  return data;
}