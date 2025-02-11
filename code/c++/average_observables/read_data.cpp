#include <DataFrame/DataFrame.h>
#include <filesystem>
#include <string>
#include <vector>

std::vector<std::string> parse_line(std::string &line) {
  std::vector<std::string> result;
  std::string delimiter = " ";
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

hmdf::StdDataFrame<unsigned long> read_csv(std::string file_path, int row_num,
                                           int last_conf) {
  hmdf::StdDataFrame<unsigned long> df;
  std::ifstream file_stream(file_path);
  std::string line;
  std::vector<std::string> parsed_line;
  std::vector<int> x_col;
  std::vector<int> y_col;
  std::vector<std::vector<double>> observables_col(20);
  std::vector<int> conf_end;
  int row_count = 0;
  while (std::getline(file_stream, line)) {
    row_count++;
    parsed_line = parse_line(line);
    if (parsed_line.size() != 22) {
      throw std::runtime_error("has wrong number of columns");
    }
    x_col.push_back(std::stoi(parsed_line[0]));
    y_col.push_back(std::stoi(parsed_line[1]));
    for (int i = 0; i < 20; i++) {
      observables_col[i].push_back(std::stod(parsed_line[i + 2]));
    }
  }
  if (row_count != row_num) {
    throw std::runtime_error("has wrong number of rows");
  }
  std::vector<unsigned long> idx_col(row_count);
  std::iota(std::begin(idx_col), std::end(idx_col), 0);
  conf_end = std::vector<int>(
      row_count, std::get<1>(get_file_conf_range(file_path)) + last_conf);
  df_load(df, idx_col, x_col, y_col, observables_col, conf_end);
  return df;
}

hmdf::StdDataFrame<unsigned long> read_data(std::string dir_path, int row_num) {
  std::string file_path;
  hmdf::StdDataFrame<unsigned long> df;
  hmdf::StdDataFrame<unsigned long> df_tmp;
  std::vector<unsigned long> idx_col;
  std::vector<int> x_col;
  std::vector<int> y_col;
  std::vector<std::vector<double>> observables_col(20);
  std::vector<int> conf_end;
  int last_conf = 0;
  int conf_tmp = 0;
  df_load(df, idx_col, x_col, y_col, observables_col, conf_end);
  std::set<std::filesystem::path> directories = get_directories(dir_path);
  for (auto &chain_dir : directories) {
    std::set<std::filesystem::path> files = get_files(chain_dir);
    last_conf = conf_tmp;
    for (auto &file_path : files) {
      try {
        df_tmp = read_csv(file_path, row_num, last_conf);
        df = df.concat<decltype(df_tmp), int, double>(df_tmp);
        conf_tmp = std::get<1>(get_file_conf_range(file_path));
      } catch (...) {
      }
    }
  }
  return df;
}