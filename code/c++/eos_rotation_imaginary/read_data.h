#pragma once

#include <DataFrame/DataFrame.h>
#include <string>

hmdf::StdDataFrame<unsigned long> read_data(std::string dir_path, int row_num,
                                            int &block_size);
hmdf::StdDataFrame<unsigned long>
read_data_polyakov(std::string dir_path, int row_num, int &block_size);
std::vector<std::vector<std::vector<std::vector<double>>>>
read_data_to_vector(std::string dir_path, int Ns, int &block_size,
                    int thermalization_length);
std::vector<std::vector<std::vector<std::vector<double>>>>
read_data_to_vector_polyakov(std::string dir_path, int Ns, int &block_size,
                             int thermalization_length);
std::tuple<int, int, int> get_lattice_sizes(std::string lattice_size);
int get_thermalization_length(std::string base_path,
                              std::string spec_additional_path,
                              std::string lattice_size, std::string boundary,
                              std::string velocity, std::string beta);
int get_bin_length(std::string base_path, std::string spec_additional_path,
                   std::string lattice_size, std::string boundary,
                   std::string velocity, std::string beta);
int get_bin_length_polyakov(std::string base_path,
                            std::string spec_additional_path,
                            std::string lattice_size, std::string boundary,
                            std::string velocity, std::string beta);