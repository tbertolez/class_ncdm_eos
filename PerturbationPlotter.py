import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# THIS PLOTS OUTPUT_PERTURBATIONS FROM CLASS, E.G. CMB AND POWER SPECTRUM

plt.style.use('paper.mplstyle')
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams['figure.facecolor'] = 'white'

####################################################
# COSMIC MICROWAVE BACKGROUND PLOTS
####################################################

if True:
    cols = ['l', 'TT', 'EE', 'TE', 'BB', 'phiphi', 'TPhi', 'Ephi']
    def class_clout_to_df(file):
        with open(file, 'r+') as myfile:
            lines = []
            i = 0
            for line in myfile.readlines():
                if i < 11:
                    i += 1
                    continue
                # print(line.strip().split("\t"))
                lines.append(line.strip().split("      "))

        npbkgdata = np.array(lines, dtype=np.float64)
        bkgdata = pd.DataFrame(npbkgdata,columns=cols)
        return bkgdata

    pert_lcdm  = class_clout_to_df('output/default00_cl_lensed.dat')
    # pert_nufld = class_clout_to_df('class_nufld/output/default_nufld_cl_lensed.dat')
    pert_ncdm  = class_clout_to_df('output/default_ncdm_0.3eV_17_cl_lensed.dat')

    # # pert_nofla = class_clout_to_df('output/no_fla_cl_lensed.dat')
    # # pert_class = class_clout_to_df('output/class_fla_cl_lensed.dat')
    # # pert_caio  = class_clout_to_df('output/caio_fla_cl_lensed.dat')


    # # DIFFERENCE TO LAMBDA-CDM PLOTS
    # # --------------------------------------------------

    fig_cl, axs_cl = plt.subplots(ncols = 1, nrows = 2, figsize = (6,9))
    for ax_cl in axs_cl:
        ax_cl.hlines(0.0,2,2500, ls = '--', lw = 1, color = 'gray')
        # ax_cl.plot(pert_nufld['l'], pert_nufld['l']*(pert_nufld['l']+1)/(2*np.pi)*pert_nufld['TT'], color = 'c', ls = '--', lw = 2, label = r'Continuity equation')
        ax_cl.plot(pert_ncdm['l'], pert_ncdm['l']*(pert_ncdm['l']+1)/(2*np.pi)*pert_ncdm['TT'], color = 'r', ls = '--', lw = 2, label = r'Full Boltzmann tower')
        ax_cl.plot(pert_lcdm['l'], pert_lcdm['l']*(pert_lcdm['l']+1)/(2*np.pi)*pert_lcdm['TT'], color = 'k', ls = '--', lw = 1, label = r'LCDM')
        # ax_cl.plot(pert_nufld['l'], (pert_nufld['TT']-pert_lcdm['TT'])/pert_lcdm['TT'], color = 'k', lw = 3, label = r'nufld')
        # ax_cl.plot(pert_ncdm['l'], (pert_ncdm['TT']-pert_lcdm['TT'])/pert_lcdm['TT'], color = 'r', ls = '--', lw = 2, label = r'ncdm')
        ax_cl.set_ylabel(r'$(C_\ell^\mathrm{TT}-C_{\ell,\mathrm{exact}}^\mathrm{TT})/C_{\ell,\mathrm{exact}}^\mathrm{TT}$')
        ax_cl.set_ylabel(r'$\frac{l(l+1)}{2\pi}C_\ell^\mathrm{TT}$')
        ax_cl.set_xlabel(r'$\ell$')
        ax_cl.set_xlim(2,2500)
        ax_cl.legend(fontsize = 12)

    axs_cl[0].set_xscale('log')
    axs_cl[1].set_xscale('linear')
    # ax_cl.set_xscale('log')

    # axs_cl[0].set_ylim(-1e-7,1e-7)
    # axs_cl[1].set_ylim(-0.2,0.2) # 1 eV
    # axs_cl[0].set_ylim(-0.05,0.05) # 0.1 eV
    # axs_cl[1].set_ylim(-0.05,0.05) # 0.1 eV
    # axs_cl[1].set_ylim(-0.002,0.002) # 0.01 eV

    fig_cl.tight_layout()
    fig_cl.savefig("comparison_cl_lensed_TT.png")

    fig_cl, axs_cl = plt.subplots(ncols = 1, nrows = 2, figsize = (6,9))
    # axs_cl[0].plot(pert_nufld['l'], pert_nufld['l']*(pert_nufld['l']+1)/(2*np.pi)*pert_nufld['TT'], color = 'c', ls = '--', lw = 2, label = r'Continuity equation')
    axs_cl[0].plot(pert_ncdm['l'], pert_ncdm['l']*(pert_ncdm['l']+1)/(2*np.pi)*pert_ncdm['TT'], color = 'r', ls = '--', lw = 2, label = r'Full Boltzmann tower')
    axs_cl[0].plot(pert_lcdm['l'], pert_lcdm['l']*(pert_lcdm['l']+1)/(2*np.pi)*pert_lcdm['TT'], color = 'k', ls = '--', lw = 1, label = r'LCDM')
    axs_cl[0].set_ylabel(r'$\frac{l(l+1)}{2\pi}C_\ell^\mathrm{TT}$')
    # axs_cl[0].set_ylim(0,1e-6)

    # axs_cl[1].plot(pert_nufld['l'], (pert_nufld['TT']-pert_lcdm['TT'])/pert_lcdm['TT'], color = 'k', lw = 3, label = r'nufld')
    axs_cl[1].plot(pert_ncdm['l'], (pert_ncdm['TT']-pert_lcdm['TT'])/pert_lcdm['TT'], color = 'r', ls = '--', lw = 2, label = r'ncdm')
    axs_cl[1].set_ylabel(r'$(C_\ell^\mathrm{TT}-C_{\ell,\mathrm{exact}}^\mathrm{TT})/C_{\ell,\mathrm{exact}}^\mathrm{TT}$')
    axs_cl[1].hlines(0.0,2,2500, ls = '--', lw = 1, color = 'gray')


    for ax_cl in axs_cl:
        # ax_cl.plot(pert_nufld['l'], pert_nufld['l']*(pert_nufld['l']+1)/(2*np.pi)*pert_nufld['TT'], color = 'c', ls = '--', lw = 2, label = r'Continuity equation')
        # ax_cl.plot(pert_ncdm['l'], pert_ncdm['l']*(pert_ncdm['l']+1)/(2*np.pi)*pert_ncdm['TT'], color = 'r', ls = '--', lw = 2, label = r'Full Boltzmann tower')
        # ax_cl.plot(pert_lcdm['l'], pert_lcdm['l']*(pert_lcdm['l']+1)/(2*np.pi)*pert_lcdm['TT'], color = 'k', ls = '--', lw = 1, label = r'LCDM')
        # ax_cl.plot(pert_nufld['l'], (pert_nufld['TT']-pert_lcdm['TT'])/pert_lcdm['TT'], color = 'k', lw = 3, label = r'nufld')
        # ax_cl.plot(pert_ncdm['l'], (pert_ncdm['TT']-pert_lcdm['TT'])/pert_lcdm['TT'], color = 'r', ls = '--', lw = 2, label = r'ncdm')
        # ax_cl.set_ylabel(r'$(C_\ell^\mathrm{TT}-C_{\ell,\mathrm{exact}}^\mathrm{TT})/C_{\ell,\mathrm{exact}}^\mathrm{TT}$')
        # ax_cl.set_ylabel(r'$\frac{l(l+1)}{2\pi}C_\ell^\mathrm{TT}$')
        ax_cl.set_xlabel(r'$\ell$')
        # ax_cl.set_xlim(2,2500)
        ax_cl.legend(fontsize = 12)
        ax_cl.set_xscale('log')
        ax_cl.set_xlim(2,2500)

    # axs_cl[1].set_xscale('log')
    # axs_cl[1].set_xscale('linear')
    # ax_cl.set_xscale('log')

    # axs_cl[0].set_ylim(-1e-7,1e-7)
    # axs_cl[1].set_ylim(-0.2,0.2) # 1 eV
    # axs_cl[0].set_ylim(-0.05,0.05) # 0.1 eV
    # axs_cl[1].set_ylim(-0.05,0.05) # 0.1 eV
    # axs_cl[1].set_ylim(-0.002,0.002) # 0.01 eV

    fig_cl.tight_layout()
    fig_cl.savefig("zoomed_cl_lensed_TT.png")

    # PLOTTING THE CMB ANISOTROPIES AS IS
    # -----------------------------------------------------

    # fig_cl, ax_cl = plt.subplots(figsize = (6,5))
    # ax_cl.hlines(0.0,2,2500, ls = '--', lw = 1, color = 'gray')
    # ax_cl.plot(pert_lcdm['l'], pert_lcdm['l']*(pert_lcdm['l']+1)/(2*np.pi)*pert_nofla['TT'], color = 'gray', lw = 3, label = r'LCDM')
    # ax_cl.plot(pert_nofla['l'], pert_nofla['l']*(pert_nofla['l']+1)/(2*np.pi)*pert_nofla['TT'], color = 'k', lw = 3, label = r'Exact')
    # ax_cl.plot(pert_class['l'], pert_class['l']*(pert_class['l']+1)/(2*np.pi)*pert_class['TT'], color = 'r', ls = '--', lw = 2, label = r'CLASS')
    # ax_cl.plot(pert_caio['l'], pert_caio['l']*(pert_caio['l']+1)/(2*np.pi)*pert_caio['TT'], color = 'c', ls = '--', lw = 2, label = r'Caio')
    # ax_cl.set_ylabel(r'$\frac{l(l+1)}{2\pi}C_\ell^\mathrm{TT}$')
    # ax_cl.set_xlabel(r'$\ell$')
    # ax_cl.set_xlim(2,2500)
    # ax_cl.legend(fontsize = 12)

    # axs_cl[0].set_xscale('log')
    # axs_cl[1].set_xscale('linear')
    # # ax_cl.set_xscale('log')

    # # axs_cl[1].set_ylim(-0.2,0.2) # 1 eV
    # # axs_cl[1].set_ylim(-0.03,0.03) # 0.1 eV
    # # axs_cl[1].set_ylim(-0.002,0.002) # 0.01 eV

    # fig_cl.tight_layout()
    # fig_cl.savefig("fla_cl_lensed_TT.png")


#######################################################
#               POWER SPECTRUM PLOTS                  #
#######################################################

cols = ['k', 'P']
def class_pkout_to_df(file):
    with open(file, 'r+') as myfile:
        lines = []
        i = 0
        for line in myfile.readlines():
            if i < 4:
                i += 1
                continue
            # print(line.strip().split("\t"))
            lines.append(line.strip().split("      "))

    npbkgdata = np.array(lines, dtype=np.float64)
    bkgdata = pd.DataFrame(npbkgdata,columns=cols)
    return bkgdata

lcdm_pk  = class_pkout_to_df('output/default00_pk.dat')
# nufld_pk = class_pkout_to_df('class_nufld/output/default_nufld_pk.dat')
ncdm_pk  = class_pkout_to_df('output/default_ncdm_0.3eV_17_pk.dat')

# caio_pk = class_pkout_to_df("output/caio_fla_pk.dat")
# clfla_pk = class_pkout_to_df("output/class_fla_pk.dat")
# nofla_pk = class_pkout_to_df("output/no_fla_pk.dat")
# lcdm_pk = class_pkout_to_df("output/lcdm_pk.dat")

# PLOTTING THE POWER SPECTRUM AND THE DIFFERENCE TO THE EXACT COMPUTATION
# -----------------------------------------------------------------------

fig_pk, axs_pk = plt.subplots(ncols = 1, nrows = 2, figsize = (6,9))
# print(caio_pk['k'], caio_pk['P'])
axs_pk[0].set_xscale('log')
axs_pk[0].set_yscale('log') 
axs_pk[0].set_xlim((1e-3,0.5))
# axs_pk[0].set_ylim((3e2,3.5e4))
# axs_pk[0].plot(nufld_pk['k'], nufld_pk['P'], label = "nufld")
axs_pk[0].plot( ncdm_pk['k'],  ncdm_pk['P'], label = "ncdm")
axs_pk[0].plot( lcdm_pk['k'],  lcdm_pk['P'], label = "lcdm")
# axs_pk[1].plot(nufld_pk['k'], nufld_pk['P']/lcdm_pk['P']-1, lw = 3, label = "Full Boltzmann tower")
axs_pk[1].plot(ncdm_pk['k'], ncdm_pk['P']/lcdm_pk['P']-1, lw = 3, label = "Continuity equation")
axs_pk[0].set_ylabel(r'$P(k)$')
axs_pk[1].set_ylabel(r'$P(k)/P_{\Lambda\mathrm{CDM}}(k)-1$')
axs_pk[0].set_xlabel(r'$k\ (h/\mathrm{Mpc})$')
axs_pk[1].set_xlabel(r'$k\ (h/\mathrm{Mpc})$')
axs_pk[1].set_xlim((1e-3,0.5))
axs_pk[1].set_ylim((-1,0.0))
axs_pk[1].set_xscale('log')
axs_pk[0].legend()
axs_pk[1].legend(fontsize = 16)
fig_pk.tight_layout()
fig_pk.savefig('power_spectrum.png')

exit()
######################################################
#             TRANSFER FUNCTIONS PLOTS               #
######################################################


cols = ['k', 'd_g', 'd_b', 'd_cdm', 'd_ur', 'd_ncdm','d_m','d_tot','phi','psi','t_g','t_b', 't_ur', 't_ncdm', 't_tot']
def class_tkout_to_df(file):
    with open(file, 'r+') as myfile:
        lines = []
        i = 0
        for line in myfile.readlines():
            if i < 11:
                i += 1
                continue
            # print(line.strip().split("\t"))
            lines.append(line.strip().split("      "))

    nptkdata = np.array(lines, dtype=np.float64)
    tkdata = pd.DataFrame(nptkdata,columns=cols)
    return tkdata

caio_tk = class_tkout_to_df("output/caio_fla_tk.dat")
clfla_tk = class_tkout_to_df("output/class_fla_tk.dat")
nofla_tk = class_tkout_to_df("output/no_fla_tk.dat")
lcdm_tk  = class_tkout_to_df("output/lcdm_tk.dat")
# caio_1eV_tk = class_tkout_to_df("output/caio_fla_1eV_tk.dat")
# clfla_1eV_tk = class_tkout_to_df("output/class_fla_1eV_tk.dat")
# nofla_1eV_tk = class_tkout_to_df("output/no_fla_1eV_tk.dat")
# lcdm_tk = class_tkout_to_df("output/lcdm_tk.dat")

# DENSITY CONTRAST TRANSFER FUNCTIONS
# --------------------------------------

fig_dtk, axs_dtk = plt.subplots(ncols = 1, nrows = 2, figsize = (6,9))
axs_dtk[0].set_xlim((5e-3,1.0))
axs_dtk[1].set_xlim((5e-3,1.0))
# axs_dtk[0].plot(caio_tk['k'], caio_tk['d_ncdm'], label = "Caio")
# axs_dtk[0].plot(clfla_tk['k'], clfla_tk['d_ncdm'], label = "CLASS fla")
# axs_dtk[0].plot(nofla_tk['k'], nofla_tk['d_ncdm'], label = "Exact")
# axs_dtk[1].plot(caio_tk['k'], caio_tk['d_tot'], label = "Caio")
# axs_dtk[1].plot(clfla_tk['k'], clfla_tk['d_tot'], label = "CLASS fla")
# axs_dtk[1].plot(nofla_tk['k'], nofla_tk['d_tot'], label = "Exact")
axs_dtk[0].plot(caio_tk['k'], caio_tk['d_ncdm']/nofla_tk['d_ncdm']-1, label = "Caio")
axs_dtk[0].plot(clfla_tk['k'], clfla_tk['d_ncdm']/nofla_tk['d_ncdm']-1, label = "CLASS fla")
axs_dtk[0].plot(nofla_tk['k'], nofla_tk['d_ncdm']/nofla_tk['d_ncdm']-1, label = "Exact", color = 'gray', linestyle = 'dashed')
axs_dtk[1].plot(caio_tk['k'], caio_tk['d_tot']/nofla_tk['d_tot']-1, label = "Caio")
axs_dtk[1].plot(clfla_tk['k'], clfla_tk['d_tot']/nofla_tk['d_tot']-1, label = "CLASS fla")
axs_dtk[1].plot(nofla_tk['k'], nofla_tk['d_tot']/nofla_tk['d_tot']-1, label = "Exact", color = 'gray', linestyle = 'dashed')
# axs_dtk[0].set_ylabel(r'$\delta_\nu$')
axs_dtk[0].set_xlabel(r'$k\ (h/\mathrm{Mpc})$')
axs_dtk[1].set_ylabel(r'$\delta_{\mathrm{tot}}/\delta_{\mathrm{tot, exact}}-1$')
axs_dtk[0].set_ylabel(r'$\delta_\nu/\delta_{\nu,\mathrm{exact}}-1$')
axs_dtk[1].set_xlabel(r'$k\ (h/\mathrm{Mpc})$')
axs_dtk[0].set_xscale('log')
axs_dtk[1].set_xscale('log')
axs_dtk[0].legend()
axs_dtk[1].legend()
fig_dtk.tight_layout()
fig_dtk.savefig('delta_transfer.png')

# DENSITY CONTRAST TRANSFER FUNCTIONS
# --------------------------------------

fig_dtk, axs_dtk = plt.subplots(ncols = 1, nrows = 2, figsize = (6,9))
axs_dtk[0].set_xlim((5e-3,1.0))
axs_dtk[1].set_xlim((5e-3,1.0))
axs_dtk[0].plot(caio_tk['k'], caio_tk['d_ncdm'], label = "Caio")
axs_dtk[0].plot(clfla_tk['k'], clfla_tk['d_ncdm'], label = "CLASS fla")
axs_dtk[0].plot(nofla_tk['k'], nofla_tk['d_ncdm'], label = "Exact")
axs_dtk[0].plot(lcdm_tk['k'], lcdm_tk['d_ncdm'], label = "lcdm")
axs_dtk[1].plot(caio_tk['k'], caio_tk['d_tot'], label = "Caio")
axs_dtk[1].plot(clfla_tk['k'], clfla_tk['d_tot'], label = "CLASS fla")
axs_dtk[1].plot(nofla_tk['k'], nofla_tk['d_tot'], label = "Exact")
axs_dtk[1].plot(lcdm_tk['k'], lcdm_tk['d_tot'], label = "lcdm")
# axs_dtk[0].plot(caio_tk['k'], caio_tk['d_ncdm']/lcdm_tk['d_ncdm']-1, label = "Caio")
# axs_dtk[0].plot(clfla_tk['k'], clfla_tk['d_ncdm']/lcdm_tk['d_ncdm']-1, label = "CLASS fla")
# axs_dtk[0].plot(nofla_tk['k'], nofla_tk['d_ncdm']/lcdm_tk['d_ncdm']-1, label = "Exact", color = 'gray', linestyle = 'dashed')
# axs_dtk[1].plot(caio_tk['k'], caio_tk['d_tot']/lcdm_tk['d_tot']-1, label = "Caio")
# axs_dtk[1].plot(clfla_tk['k'], clfla_tk['d_tot']/lcdm_tk['d_tot']-1, label = "CLASS fla")
# axs_dtk[1].plot(nofla_tk['k'], nofla_tk['d_tot']/lcdm_tk['d_tot']-1, label = "Exact", color = 'gray', linestyle = 'dashed')
axs_dtk[0].set_ylabel(r'$\delta_\nu$')
axs_dtk[1].set_ylabel(r'$\delta_{\mathrm{tot}}$')
axs_dtk[0].set_xlabel(r'$k\ (h/\mathrm{Mpc})$')
# axs_dtk[1].set_ylabel(r'$\delta_{\mathrm{tot}}/\delta_{\mathrm{tot, lcdm}}-1$')
# axs_dtk[0].set_ylabel(r'$\delta_\nu/\delta_{\nu,\mathrm{lcdm}}-1$')
axs_dtk[1].set_xlabel(r'$k\ (h/\mathrm{Mpc})$')
axs_dtk[0].set_xscale('log')
axs_dtk[1].set_xscale('log')
axs_dtk[0].legend()
axs_dtk[1].legend()
fig_dtk.tight_layout()
fig_dtk.savefig('delta_transfer_lcdm.png')


# VELOCITY DIVERGENCE TRANSFER FUNCTIONS
# -------------------------------------------

fig_ttk, axs_ttk = plt.subplots(ncols = 1, nrows = 2, figsize = (6,9))
axs_ttk[0].set_xlim((5e-3,1.0))
axs_ttk[1].set_xlim((5e-3,1.0))
axs_ttk[0].plot(caio_tk['k'], caio_tk['t_ncdm'], label = "Caio")
axs_ttk[0].plot(clfla_tk['k'], clfla_tk['t_ncdm'], label = "CLASS fla")
axs_ttk[0].plot(nofla_tk['k'], nofla_tk['t_ncdm'], label = "Exact")
axs_ttk[1].plot(caio_tk['k'], caio_tk['t_tot'], label = "Caio")
axs_ttk[1].plot(clfla_tk['k'], clfla_tk['t_tot'], label = "CLASS fla")
axs_ttk[1].plot(nofla_tk['k'], nofla_tk['t_tot'], label = "Exact")
axs_ttk[0].set_ylabel(r'$\theta_\nu$')
axs_ttk[0].set_xlabel(r'$k\ (h/\mathrm{Mpc})$')
axs_ttk[1].set_ylabel(r'$\theta_{\mathrm{tot}}$')
axs_ttk[1].set_xlabel(r'$k\ (h/\mathrm{Mpc})$')
axs_ttk[0].set_xscale('log')
axs_ttk[1].set_xscale('log')
axs_ttk[0].legend()
axs_ttk[1].legend()
fig_ttk.tight_layout()
fig_ttk.savefig('theta_transfer.png')