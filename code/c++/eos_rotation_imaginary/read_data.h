#pragma once

#include <DataFrame/DataFrame.h>
#include <string>

hmdf::StdDataFrame<unsigned long> read_data(std::string dir_path, int row_num);