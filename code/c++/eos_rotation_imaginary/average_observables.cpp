#include "data_processing.h"
#include "make_observables.h"
#include "read_data.h"
#include "visitors.h"

#include "omp.h"
#include <DataFrame/DataFrame.h>
#include <DataFrame/Utils/Threads/ThreadGranularity.h>

int main() {
  double start_time;
  double end_time;
  double search_time;

  std::string dir_path =
      "/home/ilya/soft/lattice/observables/data/eos_rotation_imaginary/"
      "5x30x121sq/OBCb_cV/0.400000v/3.8800";

  hmdf::ThreadGranularity::set_optimum_thread_level();
  // hmdf::ThreadGranularity::set_thread_level(1);

  int Ns = 121;
  int coord_max = Ns / 2;

  start_time = omp_get_wtime();
  hmdf::StdDataFrame<unsigned long> df = read_data(dir_path, Ns * Ns);
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

  start_time = omp_get_wtime();
  make_observables(df);
  end_time = omp_get_wtime();
  search_time = end_time - start_time;
  std::cout << "make_observables time: " << search_time << std::endl;

  df.remove_column<double>("2");
  df.remove_column<double>("3");
  df.remove_column<double>("4");
  df.remove_column<double>("5");
  df.remove_column<double>("6");
  df.remove_column<double>("7");
  df.remove_column<double>("8");
  df.remove_column<double>("9");
  df.remove_column<double>("10");
  df.remove_column<double>("11");
  df.remove_column<double>("12");
  df.remove_column<double>("13");
  df.remove_column<double>("14");
  df.remove_column<double>("15");
  df.remove_column<double>("16");
  df.remove_column<double>("17");
  df.remove_column<double>("18");
  df.remove_column<double>("20");
  df.remove_column<double>("21");

  RadiusSquaredVisitor<int> rad_squared_visitor(coord_max);
  df.single_act_visit<int, int>("x", "y", rad_squared_visitor);
  df.load_result_as_column(rad_squared_visitor, "rad_sqared");
  std::vector<int> radii_sq;
  std::vector<std::vector<double>> data_aver;
  hmdf::StdDataFrame<unsigned long> df1;
  hmdf::StdDataFrame<unsigned long> df_aver;
  // for (int cut = 0; cut < coord_max - 2; cut++) {
  for (int cut = 0; cut < 2; cut++) {
    start_time = omp_get_wtime();
    auto functor_cut = [coord_max, cut](const unsigned long &, const int &x,
                                        const int &y) -> bool {
      return (x <= 2 * coord_max - cut) && (x >= cut) &&
             (y <= 2 * coord_max - cut) && (y >= cut);
    };
    df = df.get_data_by_sel<int, int, decltype(functor_cut), int, double>(
        "x", "y", functor_cut);
    end_time = omp_get_wtime();
    search_time = end_time - start_time;
    std::cout << "box cut time: " << search_time << std::endl;

    radii_sq = get_radii_sq(coord_max - cut);
    for (int rad_cut : radii_sq) {
      std::cout << rad_cut << std::endl;
      start_time = omp_get_wtime();
      auto functor_rad_cut = [rad_cut](const unsigned long &,
                                       const int &rad_sq) -> bool {
        return rad_sq <= rad_cut;
      };
      df1 = df.get_data_by_sel<int, decltype(functor_rad_cut), int, double>(
          "rad_sqared", functor_rad_cut);
      end_time = omp_get_wtime();
      search_time = end_time - start_time;
      std::cout << "radii cut time: " << search_time << std::endl;

      start_time = omp_get_wtime();
      data_aver = observables_aver(df1);
      end_time = omp_get_wtime();
      search_time = end_time - start_time;
      std::cout << "observables_aver time: " << search_time << std::endl;

      std::cout.precision(10);
      start_time = omp_get_wtime();
      std::map<std::string, std::tuple<double, double>> result =
          jackknife(data_aver, 5);
      end_time = omp_get_wtime();
      search_time = end_time - start_time;
      for (auto &res : result) {
        std::cout << res.first << " " << std::get<0>(res.second) << " "
                  << std::get<1>(res.second) << std::endl;
      }
      std::cout << "jackknife time: " << search_time << std::endl;
    }
  }
}