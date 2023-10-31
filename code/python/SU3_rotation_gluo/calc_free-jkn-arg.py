import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
import mylib as lib_roenko
import lib_sychev


def main():
    matplotlib.use('Agg')
    print('Plot Pl-T, chi-T from avPL')

    parser = argparse.ArgumentParser(description='twist averapePL.txt')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Path to log directory.')
    parser.add_argument('--bin_size', type=int, required=False,
                        help='size of the bin for jacknife')
    args, _ = parser.parse_known_args()

    print(args.bin_size)
    if args.bin_size is None:
        print('bin_size is set to be equal to', 50)
        BINSIZE = 50
    else:
        BINSIZE = args.bin_size
        print('bin_size is equal to', args.bin_size)

    omega_dir_path = Path(args.log_dir)
    omega_dir_path = omega_dir_path.resolve()
    print(omega_dir_path)
    os.chdir(omega_dir_path)
    print(omega_dir_path.parts[-3])  # lattice size subdir_name

    _kx = []
    for k in range(len(omega_dir_path.parts[-3])):
        if omega_dir_path.parts[-3][k] not in '0123456789':
            print(k, omega_dir_path.parts[-3][:k])
            _kx.append(k)
    print(_kx)
    Lt = int(omega_dir_path.parts[-3][:_kx[0]])
    Lz = int(omega_dir_path.parts[-3][(_kx[0] + 1):_kx[1]])
    Ls = int(omega_dir_path.parts[-3][(_kx[1] + 1):_kx[2]])
    print(Lt, Lz, Ls, Lz * Ls * Ls)
    vol = Lz * Ls * Ls

    tempar = []
    betaar = []

    actS = []
    actSplt = []
    actSpls = []

    errS = []
    errSplt = []
    errSpls = []

    actSpl = []
    errSpl = []
    actSch = []
    errSch = []

    chiS = []
    chiSplt = []
    chiSpls = []

    errChiS = []
    errChiSplt = []
    errChiSpls = []

    PLOTERRORSBIN = True
    data_to_plot_counter = 0

    plt.clf()
    fig, ax = plt.subplots(ncols=2, constrained_layout=True)

    for beta_dir_path in sorted(omega_dir_path.iterdir()):
        if not beta_dir_path.is_dir():
            continue
        filename = beta_dir_path / 'actionShort.txt'
        print(filename)
        if not filename.exists():
            continue
        data = np.genfromtxt(str(filename))  # merged action data
        print(data[0])
        averSplt = np.zeros(len(data), dtype=(float, float))
        averSpls = np.zeros(len(data), dtype=(float, float))
        average_S_data = np.zeros(len(data), dtype=(float, float))
        averSpl = np.zeros(len(data), dtype=(float, float))
        averSch = np.zeros(len(data), dtype=(float, float))
        for i in range(len(data)):
            # averPL[i] = m.fabs(data[i][0])
            average_S_data[i] = data[i][3]
            averSpl[i] = data[i][4]
            averSpls[i] = data[i][5]
            averSplt[i] = data[i][6]
            averSch[i] = data[i][7]

        # mean, dev, nobiasedmean, bias = lib_roenko.jackknifeMy(averPL)
        # mean2, dev2, nobiasedmean2, bias2 = lib_roenko.jackknifeMy(averPL2)
        # plA.append(mean)
        # plAerr.append(dev)
        # chiA.append(mean2 - mean*mean)
        # chiAerr.append(dev2 + 2*mean*dev)

        if PLOTERRORSBIN:
            if data_to_plot_counter < 15:
                print(data_to_plot_counter)

                min_bin_size = 1
                max_bin_size = len(average_S_data) // 5
                bin_size_step_factor = 1.05

                bin_sizes_to_plot = lib_sychev.int_log_range(
                    min_bin_size, max_bin_size, bin_size_step_factor)

                sychev_results = np.array([
                    lib_sychev.bootstrap_meanstd_handler(
                        average_S_data, bin_size, lib_sychev.mean)
                    for bin_size in bin_sizes_to_plot])

                # errorsbin = np.array([lib_roenko.jackknifeBin3(average_S_data, bin_size) for bin_size in bin_sizes_to_plot])

                ax[0].errorbar(bin_sizes_to_plot, sychev_results[:, 1],
                               label='sychev')
                ax[0].set_xscale('log')

                # ax[1].errorbar([bin_size for bin_size in bin_sizes_to_plot], errorsbin[:, 1],
                #                label='roenko')
                # ax[1].set_xscale('log')

                # Archived version
                # errorsbin = np.array([lib_roenko.jackknifeBin3(average_S_data, bin_size) for bin_size in range(1, 500, 4)])
                # ax[0].errorbar([bin_size for bin_size in range(1, 500, 4)], errorsbin[:, 1],
                #                label=str(beta_dir_path.parts[-1]))
                # ax[1].errorbar([bin_size for bin_size in range(1, 500, 4)], vol * errorsbin[:, 3],
                #                label=str(beta_dir_path.parts[-1]))
                data_to_plot_counter += 1
            print('PLOTERRORSBIN')

        bracketing_factor = 1.01
        bracketing_number = 11
        binsizes = [
            int(BINSIZE * bracketing_factor**i)
            for i in range(-bracketing_number//2, bracketing_number//2 + 1)
        ]
        mean = []
        dev = []
        meanS = []
        devS = []
        for binsize in binsizes:
            _mean, _dev, _meanS, _devS = lib_roenko.jackknifeBin3(average_S_data, binsize)
            mean.append(_mean)
            dev.append(_dev)
            meanS.append(_meanS)
            devS.append(_devS)
        mean = np.array(mean)
        dev = np.array(dev)
        meanS = np.array(meanS)
        devS = np.array(devS)

        actS.append(mean.mean())
        errS.append(dev.mean())
        chiS.append(vol * meanS.mean())
        errChiS.append(vol * devS.mean())

        mean, dev, meanS, devS = lib_roenko.jackknifeBin3(averSplt, BINSIZE)
        actSplt.append(mean)
        errSplt.append(dev)
        chiSplt.append(vol * meanS)
        errChiSplt.append(vol * devS)

        mean, dev, meanS, devS = lib_roenko.jackknifeBin3(averSpls, BINSIZE)
        actSpls.append(mean)
        errSpls.append(dev)
        chiSpls.append(vol * meanS)
        errChiSpls.append(vol * devS)

        mean, dev, meanS, devS = lib_roenko.jackknifeBin3(averSpl, BINSIZE)
        actSpl.append(mean)
        errSpl.append(dev)

        mean, dev, meanS, devS = lib_roenko.jackknifeBin3(averSch, BINSIZE)
        actSch.append(mean)
        errSch.append(dev)

        tempar.append(lib_roenko.get_temperature(float(beta_dir_path.parts[-1]), Lt))
        betaar.append(float(beta_dir_path.parts[-1]))

    # ax[1].grid(True)
    ax[0].grid(True)
    ax[0].legend()
    # ax[1].legend()
    ax[0].set_xlabel('bin_size')
    # ax[1].set_xlabel(r'$B$')
    # ax[1].set_ylabel(r'$\delta act_\chi$')
    ax[0].set_ylabel('S_error')
    plt.title(f"{len(average_S_data)} {beta_dir_path.parts[-4]} "
              f"{beta_dir_path.parts[-2]} {beta_dir_path.parts[-3]}")
    fig.set_size_inches(12.8, 4.8)
    if PLOTERRORSBIN:
        fig.savefig('errS.jpg')
    plt.close()

    print(betaar)
    print(actS)
    print(actSpls)
    print(actSplt)

    # def fit_func_gauss(T, a0, T0, dT):
    #     return a0 * np.exp(-(T - T0) ** 2 / (2. * dT ** 2))
    # def fit_func_atan(T, a0, a1, T0, dT):
    #     return a0 + a1 * np.arctan((T - T0) / dT)
    # def fit_func_lin(T, a0, a1, T0):
    #     return a0 + a1 * (T - T0) * (T - T0)

    plt.clf()
    plt.errorbar(betaar, actS, fmt='ro', yerr=errS, ecolor='darkred', capsize=3, ms=5)
    plt.grid(True)
    plt.xlabel(r'$\beta$')
    plt.title(r'$\Omega = $' + beta_dir_path.parts[-2] + ' ' + beta_dir_path.parts[-3])
    plt.ylabel(r'$S_g/\beta$')
    plt.savefig('actS.jpg')

    np.savetxt("beta_actS.txt", lib_roenko.ordering(np.array([[betaar[i], actS[i], errS[i]] for i in range(len(betaar))])))
    np.savetxt("T_actS.txt", lib_roenko.ordering(np.array([[tempar[i], actS[i], errS[i]] for i in range(len(betaar))])))

    plt.clf()
    plt.errorbar(betaar, chiS, fmt='ro', yerr=errChiS, ecolor='darkred', capsize=3, ms=5)
    plt.grid(True)
    plt.xlabel(r'$\beta$')
    plt.title(r'$\Omega = $' + beta_dir_path.parts[-2] + ' ' + beta_dir_path.parts[-3])
    plt.ylabel(r'$\chi S_g/\beta$')
    plt.savefig('actChiS.jpg')

    np.savetxt("beta_chiS.txt", lib_roenko.ordering(np.array([[betaar[i], chiS[i], errChiS[i]] for i in range(len(betaar))])))
    np.savetxt("T_chiS.txt", lib_roenko.ordering(np.array([[tempar[i], chiS[i], errChiS[i]] for i in range(len(betaar))])))

    plt.clf()
    plt.errorbar(betaar, actSplt, fmt='ro', yerr=errSplt, ecolor='darkred', capsize=3, ms=5)
    plt.grid(True)
    plt.xlabel(r'$\beta$')
    plt.title(r'$\Omega = $' + beta_dir_path.parts[-2] + ' ' + beta_dir_path.parts[-3])
    plt.ylabel(r'$S_g \tau/\beta$')
    plt.savefig('actSplt.jpg')

    np.savetxt("beta_actSplt.txt", lib_roenko.ordering(np.array([[betaar[i], actSplt[i], errSplt[i]] for i in range(len(betaar))])))
    np.savetxt("T_actSplt.txt", lib_roenko.ordering(np.array([[tempar[i], actSplt[i], errSplt[i]] for i in range(len(betaar))])))

    plt.clf()
    plt.errorbar(betaar, chiSplt, fmt='ro', yerr=errChiSplt, ecolor='darkred', capsize=3, ms=5)
    plt.grid(True)
    plt.xlabel(r'$\beta$')
    plt.title(r'$\Omega = $' + beta_dir_path.parts[-2] + ' ' + beta_dir_path.parts[-3])
    plt.ylabel(r'$\chi S_g \tau/\beta$')
    plt.savefig('actChiSplt.jpg')

    np.savetxt("beta_chiSplt.txt", lib_roenko.ordering(np.array([[betaar[i], chiSplt[i], errChiSplt[i]] for i in range(len(betaar))])))
    np.savetxt("T_chiSplt.txt", lib_roenko.ordering(np.array([[tempar[i], chiSplt[i], errChiSplt[i]] for i in range(len(betaar))])))

    plt.clf()
    plt.errorbar(betaar, actSpls, fmt='ro', yerr=errSpls, ecolor='darkred', capsize=3, ms=5)
    plt.grid(True)
    plt.xlabel(r'$\beta$')
    plt.title(r'$\Omega = $' + beta_dir_path.parts[-2] + ' ' + beta_dir_path.parts[-3])
    plt.ylabel(r'$S_g \sigma/\beta$')
    plt.savefig('actSpls.jpg')

    np.savetxt("beta_actSpls.txt", lib_roenko.ordering(np.array([[betaar[i], actSpls[i], errSpls[i]] for i in range(len(betaar))])))
    np.savetxt("T_actSpls.txt", lib_roenko.ordering(np.array([[tempar[i], actSpls[i], errSpls[i]] for i in range(len(betaar))])))

    plt.clf()
    plt.errorbar(betaar, chiSpls, fmt='ro', yerr=errChiSpls, ecolor='darkred', capsize=3, ms=5)
    plt.grid(True)
    plt.xlabel(r'$\beta$')
    plt.title(r'$\Omega = $' + beta_dir_path.parts[-2] + ' ' + beta_dir_path.parts[-3])
    plt.ylabel(r'$\chi S_g \sigma/\beta$')
    plt.savefig('actChiSpls.jpg')

    np.savetxt("beta_chiSpls.txt", lib_roenko.ordering(np.array([[betaar[i], chiSpls[i], errChiSpls[i]] for i in range(len(betaar))])))
    np.savetxt("T_chiSpls.txt", lib_roenko.ordering(np.array([[tempar[i], chiSpls[i], errChiSpls[i]] for i in range(len(betaar))])))

    plt.clf()
    plt.errorbar(betaar, actSpl, fmt='ro', yerr=errSpl, ecolor='darkred', capsize=3, ms=5)
    plt.grid(True)
    plt.xlabel(r'$\beta$')
    plt.title(r'$\Omega = $' + beta_dir_path.parts[-2] + ' ' + beta_dir_path.parts[-3])
    plt.ylabel(r'$S_{g,pl}/\beta$')
    plt.savefig('actSpl.jpg')

    np.savetxt("beta_actSpl.txt", lib_roenko.ordering(np.array([[betaar[i], actSpl[i], errSpl[i]] for i in range(len(betaar))])))
    # np.savetxt("T_actS.txt", lib_roenko.ordering(np.array([[tempar[i], actS[i], errS[i]] for i in range(len(betaar))])))

    plt.clf()
    plt.errorbar(betaar, actSch, fmt='ro', yerr=errSch, ecolor='darkred', capsize=3, ms=5)
    plt.grid(True)
    plt.xlabel(r'$\beta$')
    plt.title(r'$\Omega = $' + beta_dir_path.parts[-2] + ' ' + beta_dir_path.parts[-3])
    plt.ylabel(r'$S_{g,ch}/\beta$')
    plt.savefig('actSch.jpg')

    np.savetxt("beta_actSch.txt", lib_roenko.ordering(np.array([[betaar[i], actSch[i], errSch[i]] for i in range(len(betaar))])))
    # np.savetxt("T_actS.txt", lib_roenko.ordering(np.array([[tempar[i], actS[i], errS[i]] for i in range(len(betaar))])))


if __name__ == '__main__':
    main()
