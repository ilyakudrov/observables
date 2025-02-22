#include "omp.h"
#include <DataFrame/DataFrame.h>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

std::vector<std::string> parse_line(std::string s) {
  const char delimiter = ' ';
  size_t start = 0;
  size_t end = s.find_first_of(delimiter);

  std::vector<std::string> output;

  while (end <= std::string::npos) {
    output.emplace_back(s.substr(start, end - start));

    if (end == std::string::npos)
      break;

    start = end + 1;
    end = s.find_first_of(delimiter, start);
  }

  return output;
}

std::set<std::filesystem::path> get_directories(std::string dir_path) {
  const std::filesystem::path path{dir_path};
  std::set<std::filesystem::path> directories;
  for (auto const &dir_entry : std::filesystem::directory_iterator{path}) {
    if (std::filesystem::is_directory(dir_entry.path()))
      directories.insert(dir_entry.path());
  }
  return directories;
}

bool is_observables_file(std::string file_path) {
  std::string s = file_path.substr(file_path.rfind("/") + 1);
  return (s.rfind("block", 0) == 0 && s.substr(s.find("-") + 1, 5) == "SEBxy");
}

std::set<std::filesystem::path> get_files(std::string dir_path) {
  const std::filesystem::path path{dir_path};
  std::set<std::filesystem::path> files;
  for (auto const &file_path : std::filesystem::directory_iterator{path}) {
    if (std::filesystem::is_regular_file(file_path.path()) &&
        std::filesystem::file_size(file_path.path()) > 0) {
      if (is_observables_file(file_path.path())) {
        files.insert(file_path.path());
      }
    }
  }
  return files;
}

std::tuple<int, int> get_file_conf_range(std::string file_path) {
  std::string file_name = file_path.substr(file_path.rfind("/") + 1);

  std::string a = file_name.substr(0, file_name.find("."));
  a = a.substr(a.find("_") + 1);
  return std::tuple<int, int>(std::stoi(a.substr(0, a.find("_"))),
                              std::stoi(a.substr(a.find("_") + 1)));
}

int get_block_size(std::string file_path) {
  std::string file_name = file_path.substr(file_path.rfind("/") + 1);
  return stoi(file_name.substr(5, file_name.find("-") - 5));
}

void df_load(hmdf::StdDataFrame<unsigned long> &df,
             std::vector<unsigned long> &idx_col, std::vector<int> &x_col,
             std::vector<int> &y_col,
             std::vector<std::vector<double>> &observables_col,
             std::vector<int> &conf_end) {
  std::iota(std::begin(idx_col), std::end(idx_col), 0);
  df.load_data(std::move(idx_col), std::make_pair("x", x_col),
               std::make_pair("y", y_col),
               std::make_pair("2", observables_col[0]),
               std::make_pair("3", observables_col[1]),
               std::make_pair("4", observables_col[2]),
               std::make_pair("5", observables_col[3]),
               std::make_pair("6", observables_col[4]),
               std::make_pair("7", observables_col[5]),
               std::make_pair("8", observables_col[6]),
               std::make_pair("9", observables_col[7]),
               std::make_pair("10", observables_col[8]),
               std::make_pair("11", observables_col[9]),
               std::make_pair("12", observables_col[10]),
               std::make_pair("13", observables_col[11]),
               std::make_pair("14", observables_col[12]),
               std::make_pair("15", observables_col[13]),
               std::make_pair("16", observables_col[14]),
               std::make_pair("17", observables_col[15]),
               std::make_pair("18", observables_col[16]),
               std::make_pair("S", observables_col[17]),
               std::make_pair("20", observables_col[18]),
               std::make_pair("21", observables_col[19]),
               std::make_pair("conf_end", conf_end));
}

hmdf::StdDataFrame<unsigned long>
read_csv(std::vector<int> &x_col, std::vector<int> &y_col,
         std::vector<std::vector<double>> &observables_col,
         std::vector<int> &conf_end, std::string file_path, int row_num,
         int last_conf) {
  double start_time;
  double end_time;
  double search_time;
  hmdf::StdDataFrame<unsigned long> df;
  std::ifstream file_stream(file_path);
  std::string line;
  std::vector<std::string> parsed_line;
  std::vector<int> x_col_tmp;
  x_col_tmp.reserve(row_num);
  std::vector<int> y_col_tmp;
  y_col_tmp.reserve(row_num);
  std::vector<std::vector<double>> observables_col_tmp(20);
  for (int i = 0; i < 20; i++) {
    observables_col_tmp[i].reserve(row_num);
  }
  std::vector<int> conf_end_tmp;
  conf_end_tmp.reserve(row_num);
  int row_count = 0;
  while (std::getline(file_stream, line)) {
    row_count++;
    parsed_line = parse_line(line);
    if (parsed_line.size() != 22) {
      throw std::runtime_error("has wrong number of columns");
    }
    x_col_tmp.push_back(std::stoi(parsed_line[0]));
    y_col_tmp.push_back(std::stoi(parsed_line[1]));
    for (int i = 0; i < 20; i++) {
      observables_col_tmp[i].push_back(std::stod(parsed_line[i + 2]));
    }
  }
  if (row_count != row_num) {
    throw std::runtime_error("has wrong number of rows");
  }
  conf_end_tmp = std::vector<int>(
      row_count, std::get<1>(get_file_conf_range(file_path)) + last_conf);

  x_col.insert(x_col.end(), x_col_tmp.begin(), x_col_tmp.end());
  y_col.insert(y_col.end(), y_col_tmp.begin(), y_col_tmp.end());
  for (int i = 0; i < 20; i++) {
    observables_col[i].insert(observables_col[i].end(),
                              observables_col_tmp[i].begin(),
                              observables_col_tmp[i].end());
  }
  conf_end.insert(conf_end.end(), conf_end_tmp.begin(), conf_end_tmp.end());
  return df;
}

hmdf::StdDataFrame<unsigned long> read_data(std::string dir_path, int row_num,
                                            int &block_size) {
  double start_time;
  double end_time;
  double search_time;
  std::string file_path;
  hmdf::StdDataFrame<unsigned long> df;
  std::vector<int> x_col;
  std::vector<int> y_col;
  std::vector<std::vector<double>> observables_col(20);
  std::vector<int> conf_end;
  int last_conf = 0;
  int conf_tmp = 0;
  std::set<std::filesystem::path> directories = get_directories(dir_path);
  int count = 0;
  for (auto &chain_dir : directories) {
    std::set<std::filesystem::path> files = get_files(chain_dir);
    last_conf = conf_tmp;
    for (auto &file_path : files) {
      try {
        read_csv(x_col, y_col, observables_col, conf_end, file_path, row_num,
                 last_conf);
        conf_tmp = std::get<1>(get_file_conf_range(file_path));
        if (count == 0) {
          block_size = get_block_size(file_path);
        }
      } catch (...) {
      }
    }
  }
  std::vector<unsigned long> idx_col(x_col.size());
  std::iota(std::begin(idx_col), std::end(idx_col), 0);
  df_load(df, idx_col, x_col, y_col, observables_col, conf_end);
  return df;
}

std::tuple<int, int, int> get_lattice_sizes(std::string lattice_size) {
  std::string a = lattice_size.substr(0, lattice_size.find("x"));
  std::string b =
      lattice_size.substr(lattice_size.find("x") + 1,
                          lattice_size.rfind("x") - lattice_size.find("x") - 1);
  std::string c = lattice_size.substr(lattice_size.rfind("x") + 1);
  return std::tuple<int, int, int>(std::stoi(a), std::stoi(b), std::stoi(c));
}

std::vector<std::tuple<double, int>>
read_spec_therm(std::string therm_spec_path) {
  if (!std::filesystem::exists(therm_spec_path)) {
    throw std::runtime_error(therm_spec_path + "does not exist");
  }
  std::vector<std::tuple<double, int>> result;
  std::ifstream file_stream(therm_spec_path);
  std::string line;
  std::vector<std::string> parsed_line;
  while (std::getline(file_stream, line)) {
    if (line != "\n") {
      parsed_line = parse_line(line);
      if (parsed_line.size() != 2) {
        throw std::runtime_error(therm_spec_path +
                                 "has wrong number of columns");
      }
      result.push_back(std::make_tuple(std::stod(parsed_line[0]),
                                       std::stoi(parsed_line[1])));
    }
  }
  return result;
}

bool assert_double_equal(double a, double b, double epsilon) {
  return std::abs(a - b) < epsilon;
}

int get_spec(std::vector<std::tuple<double, int>> &spec_therm, double beta) {
  for (int i = 0; i < spec_therm.size(); i++) {
    if (assert_double_equal(std::get<0>(spec_therm[i]), beta, 1e-5)) {
      return std::get<1>(spec_therm[i]);
    }
  }
  for (int i = 0; i < spec_therm.size(); i++) {
    if (assert_double_equal(std::get<0>(spec_therm[i]), 0, 1e-5)) {
      return std::get<1>(spec_therm[i]);
    }
  }
  throw std::runtime_error(std::to_string(beta) + "beta is not found");
  return -1;
}

int get_thermalization_length(std::string base_path,
                              std::string spec_additional_path,
                              std::string lattice_size, std::string boundary,
                              std::string velocity, std::string beta) {
  int therm_length = 0;
  std::stringstream ss;
  std::vector<std::tuple<double, int>> spec_therm_vec;
  try {
    ss << base_path << "/" << lattice_size << "/" << boundary << "/" << velocity
       << "/" << "spec_therm.log";
    spec_therm_vec = read_spec_therm(ss.str());
    therm_length = get_spec(spec_therm_vec, std::stod(beta));
  } catch (const char *error) {
    std::cout << error << std::endl;
    try {
      ss.str(std::string());
      ss << spec_additional_path << "/" << lattice_size << "/" << boundary
         << "/" << velocity << "/" << "spec_therm.log";
      spec_therm_vec = read_spec_therm(ss.str());
      therm_length = get_spec(spec_therm_vec, std::stod(beta));
    } catch (const char *error1) {
      std::cout << error1 << std::endl;
      throw std::runtime_error("get_thermalization_length failed");
    }
  }
  return therm_length;
}

int get_bin_length(std::string base_path, std::string spec_additional_path,
                   std::string lattice_size, std::string boundary,
                   std::string velocity, std::string beta) {
  int therm_length = 0;
  std::stringstream ss;
  std::vector<std::tuple<double, int>> spec_bin_vec;
  try {
    ss << base_path << "/" << lattice_size << "/" << boundary << "/" << velocity
       << "/" << "spec_bin_S.log";
    spec_bin_vec = read_spec_therm(ss.str());
    therm_length = get_spec(spec_bin_vec, std::stod(beta));
  } catch (const char *error) {
    std::cout << error << std::endl;
    try {
      ss.str(std::string());
      ss << spec_additional_path << "/" << lattice_size << "/" << boundary
         << "/" << velocity << "/" << "spec_bin_S.log";
      spec_bin_vec = read_spec_therm(ss.str());
      therm_length = get_spec(spec_bin_vec, std::stod(beta));
    } catch (const char *error1) {
      std::cout << error1 << std::endl;
      throw std::runtime_error("get_thermalization_length failed");
    }
  }
  return therm_length;
}