#include "make_observables.h"
#include "read_data.h"

#include "omp.h"
#include "visitors.h"
#include <DataFrame/DataFrame.h>

int main() {
  double start_time;
  double end_time;
  double search_time;

  std::string dir_path =
      "/home/ilya/soft/lattice/observables/data/eos_rotation_imaginary/"
      "5x30x121sq/OBCb_cV/0.400000v/3.8800";

  start_time = omp_get_wtime();
  hmdf::StdDataFrame<unsigned long> df = read_data(dir_path, 14641);
  end_time = omp_get_wtime();
  search_time = end_time - start_time;
  std::cout << "read time: " << search_time << std::endl;

  int thermalization_length = 50;
  auto functor_less = [thermalization_length](const unsigned long &,
                                              const int &conf_last) -> bool {
    return conf_last > thermalization_length;
  };

  start_time = omp_get_wtime();
  df = df.get_data_by_sel<int, decltype(functor_less), int, double>(
      "conf_end", functor_less);
  end_time = omp_get_wtime();
  search_time = end_time - start_time;
  std::cout << "remove thermalization time: " << search_time << std::endl;

  //   start_time = omp_get_wtime();
  //   make_observables(df);
  //   end_time = omp_get_wtime();
  //   search_time = end_time - start_time;
  //   std::cout << "make_observables time: " << search_time << std::endl;

  //   df.remove_column<double>("2");
  //   df.remove_column<double>("3");
  //   df.remove_column<double>("4");
  //   df.remove_column<double>("5");
  //   df.remove_column<double>("6");
  //   df.remove_column<double>("7");
  //   df.remove_column<double>("8");
  //   df.remove_column<double>("9");
  //   df.remove_column<double>("10");
  //   df.remove_column<double>("11");
  //   df.remove_column<double>("12");
  //   df.remove_column<double>("13");
  //   df.remove_column<double>("14");
  //   df.remove_column<double>("15");
  //   df.remove_column<double>("16");
  //   df.remove_column<double>("17");
  //   df.remove_column<double>("18");
  //   df.remove_column<double>("20");
  //   df.remove_column<double>("21");

  int coord_max = 60;
  RadiusSquaredVisitor<int> rad_squared_visitor(coord_max);
  df.single_act_visit<int, int>("x", "y", rad_squared_visitor);
  df.load_result_as_column(rad_squared_visitor, "rad_sqared");

  int cut = 0;
  auto functor_cut = [coord_max, cut](const unsigned long &, const int &x,
                                      const int &y) -> bool {
    return (x <= 2 * coord_max - cut) && (x >= cut) &&
           (y <= 2 * coord_max - cut) && (y >= cut);
  };

  start_time = omp_get_wtime();
  df = df.get_data_by_sel<int, int, decltype(functor_cut), int, double>(
      "x", "y", functor_cut);
  end_time = omp_get_wtime();
  search_time = end_time - start_time;
  std::cout << "cut time: " << search_time << std::endl;

  int rad_max = 60;
  int rad_cut = rad_max * rad_max * 2;
  //   int rad_cut = rad_max * rad_max;
  auto functor_rad_cut = [coord_max, rad_cut](const unsigned long &,
                                              const int &x,
                                              const int &y) -> bool {
    return (x - coord_max) * (x - coord_max) +
               (y - coord_max) * (y - coord_max) <=
           rad_cut;
  };

  start_time = omp_get_wtime();
  df = df.get_data_by_sel<int, int, decltype(functor_rad_cut), int, double>(
      "x", "y", functor_rad_cut);
  end_time = omp_get_wtime();
  search_time = end_time - start_time;
  std::cout << "rad cut time: " << search_time << std::endl;

  start_time = omp_get_wtime();
  //   auto result = df.groupby1<int>(
  //       "conf_end", hmdf::LastVisitor<int>(),
  //       std::make_tuple("S", "S", hmdf::MeanVisitor<double>()),
  //       std::make_tuple("Jv", "Jv", hmdf::MeanVisitor<double>()),
  //       std::make_tuple("Jv1", "Jv1", hmdf::MeanVisitor<double>()),
  //       std::make_tuple("Jv2", "Jv2", hmdf::MeanVisitor<double>()),
  //       std::make_tuple("Blab", "Blab", hmdf::MeanVisitor<double>()),
  //       std::make_tuple("E", "E", hmdf::MeanVisitor<double>()),
  //       std::make_tuple("Elab", "Elab", hmdf::MeanVisitor<double>()),
  //       std::make_tuple("Bz", "Bz", hmdf::MeanVisitor<double>()),
  //       std::make_tuple("Bxy", "Bxy", hmdf::MeanVisitor<double>()),
  //       std::make_tuple("Ez", "Ez", hmdf::MeanVisitor<double>()),
  //       std::make_tuple("Exy", "Exy", hmdf::MeanVisitor<double>()),
  //       std::make_tuple("ElabzT", "ElabzT", hmdf::MeanVisitor<double>()),
  //       std::make_tuple("ElabxyT", "ElabxyT", hmdf::MeanVisitor<double>()),
  //       std::make_tuple("Ae", "Ae", hmdf::MeanVisitor<double>()),
  //       std::make_tuple("Am", "Am", hmdf::MeanVisitor<double>()),
  //       std::make_tuple("AlabeT", "AlabeT", hmdf::MeanVisitor<double>()));
  auto result = df.groupby1<int>(
      "conf_end", hmdf::LastVisitor<int>(),
      std::make_tuple("2", "2", hmdf::MeanVisitor<double>()),
      std::make_tuple("3", "3", hmdf::MeanVisitor<double>()),
      std::make_tuple("4", "4", hmdf::MeanVisitor<double>()),
      std::make_tuple("5", "5", hmdf::MeanVisitor<double>()),
      std::make_tuple("6", "6", hmdf::MeanVisitor<double>()),
      std::make_tuple("7", "7", hmdf::MeanVisitor<double>()),
      std::make_tuple("8", "8", hmdf::MeanVisitor<double>()),
      std::make_tuple("9", "9", hmdf::MeanVisitor<double>()),
      std::make_tuple("10", "10", hmdf::MeanVisitor<double>()),
      std::make_tuple("11", "11", hmdf::MeanVisitor<double>()),
      std::make_tuple("12", "12", hmdf::MeanVisitor<double>()),
      std::make_tuple("13", "13", hmdf::MeanVisitor<double>()),
      std::make_tuple("14", "14", hmdf::MeanVisitor<double>()),
      std::make_tuple("15", "15", hmdf::MeanVisitor<double>()),
      std::make_tuple("16", "16", hmdf::MeanVisitor<double>()),
      std::make_tuple("17", "17", hmdf::MeanVisitor<double>()),
      std::make_tuple("18", "18", hmdf::MeanVisitor<double>()),
      std::make_tuple("S", "S", hmdf::MeanVisitor<double>()),
      std::make_tuple("20", "20", hmdf::MeanVisitor<double>()),
      std::make_tuple("21", "21", hmdf::MeanVisitor<double>()));
  end_time = omp_get_wtime();
  search_time = end_time - start_time;
  std::cout << "groupby time: " << search_time << std::endl;

  //   result.write<std::ostream, int, double>(std::cout,
  //   hmdf::io_format::csv2);

  start_time = omp_get_wtime();
  make_observables(result);
  end_time = omp_get_wtime();
  search_time = end_time - start_time;
  std::cout << "make_observables time: " << search_time << std::endl;

  result.remove_column<double>("2");
  result.remove_column<double>("3");
  result.remove_column<double>("4");
  result.remove_column<double>("5");
  result.remove_column<double>("6");
  result.remove_column<double>("7");
  result.remove_column<double>("8");
  result.remove_column<double>("9");
  result.remove_column<double>("10");
  result.remove_column<double>("11");
  result.remove_column<double>("12");
  result.remove_column<double>("13");
  result.remove_column<double>("14");
  result.remove_column<double>("15");
  result.remove_column<double>("16");
  result.remove_column<double>("17");
  result.remove_column<double>("18");
  result.remove_column<double>("20");
  result.remove_column<double>("21");

  // result.write<std::ostream, int, double>(std::cout,
  // hmdf::io_format::csv2);

  std::cout.precision(10);
  std::vector<std::string> observables = {
      "S",   "Jv", "Jv1", "Jv2",    "Blab",    "E",  "Elab", "Bz",
      "Bxy", "Ez", "Exy", "ElabzT", "ElabxyT", "Ae", "Am",   "AlabeT"};
  JackknifeVisitor jackknife_visitor(5);
  start_time = omp_get_wtime();
  for (auto &observable : observables) {
    result.single_act_visit<double>(observable.c_str(), jackknife_visitor);
    auto S_result = jackknife_visitor.get_result();
    std::cout << observable << ": " << std::get<0>(S_result) << " "
              << std::get<1>(S_result) << std::endl;
  }
  end_time = omp_get_wtime();
  search_time = end_time - start_time;
  std::cout << "jackknife time: " << search_time << std::endl;
}