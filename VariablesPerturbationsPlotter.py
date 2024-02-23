import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# THIS PLOTS _perturbations_kX_s FROM CLASS, E.G. variables as a function of a, tau

plt.style.use('paper.mplstyle')
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams['figure.facecolor'] = 'white'

cols_bkg_nufld = ['z','t','tau','H','x','Dang','Dlum','rs','rho_g', 'rho_b','rho_cdm', 'rho_nufld[0]', 'p_nufld[0]', 'w_nufld[0]', 'w_prime_nufld[0]', 'w_mass_nufld[0]', 'w_prime_mass_nufld[0]',
              'rho_lambda', 'rho_ur', 'rho_crit',
              'rho_tot', 'p_tot', 'p_tot_prime', 'gr.fac.D', 'gr.fac.f']
cols_bkg_ncdm = ['z','t','tau','H','x','Dang','Dlum','rs','rho_g', 'rho_b','rho_cdm', 'rho_ncdm[0]', 'p_ncdm[0]', 'rho_lambda', 'rho_ur', 'rho_crit',
        'rho_tot', 'p_tot', 'p_tot_prime', 'gr.fac.D', 'gr.fac.f']

def class_bkgout_to_df(file, cols = cols_bkg_nufld):
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

# bkg_nufld = class_bkgout_to_df('class_nufld/output/default_nufld_background.dat')
bkg_ncdm  = class_bkgout_to_df('output/default_ncdm_background.dat', cols=cols_bkg_ncdm)


cols_ncdm = ['tau', 'a', 'delta_g', 'theta_g', 'shear_g', 'pol0_g', 'pol1_g', 'pol2_g',
              'delta_b', 'theta_b', 'psi', 'phi',
              'delta_ur', 'theta_ur', 'shear_ur', 
              'delta_cdm', 'theta_cdm',
              'delta_ncdm[0]', 'theta_ncdm[0]', 'shear_ncdm[0]', 'cs2_gauge[0]', 'cs2_fluid[0]']


cols_nufld = ['tau', 'a', 'delta_g', 'theta_g', 'shear_g', 'pol0_g', 'pol1_g', 'pol2_g',
              'delta_b', 'theta_b', 'psi', 'phi',
              'delta_ur', 'theta_ur', 'shear_ur', 
              'delta_cdm', 'theta_cdm',
              'delta_nufld[0]', 'theta_nufld[0]', 'shear_nufld[0]', 'cs2_nufld[0]']

def class_pertvars_to_df(file, cols = cols_nufld):
    with open(file, 'r+') as myfile:
        lines = []
        i = 0
        for line in myfile.readlines():
            if i < 2:
                i += 1
                continue
            # print(line.strip().split("\t"))
            lines.append(line.strip().split("      "))

    npbkgdata = np.array(lines, dtype=np.float64)
    bkgdata = pd.DataFrame(npbkgdata,columns=cols)
    return bkgdata

output_k = [1e-3,1e-2]
output_k = [1e-3,1e-2,5e-2,8e-2,1e-1,2e-1,3e-1,5e-1,1e0]
mass = '0.3eV'
lmax = '100'
# output_k = [5.58480e-01, 6.27416e-01, 7.37130e-01]
# output_k = [6.27416e-01, 7.37130e-01]
# output_k = [0.0874, 0.8952, 1.1252]
# pert_nufld = {}
pert_ncdm  = {}
for i in range(len(output_k)):
    # pert_nufld.update({output_k[i]: class_pertvars_to_df('class_nufld/output/default_nufld_perturbations_k'+str(i)+'_s.dat')})
    pert_ncdm.update({output_k[i]: class_pertvars_to_df('output/default_ncdm_'+mass+'_'+lmax+'_perturbations_k'+str(i)+'_s.dat', cols = cols_ncdm)})

colors = ["#70CAD1","#7EBC89","#FE5D26"]
colors = ["#03045e","#0077b6","#0096c7","#00b4d8","#48cae4","#90e0ef","#ade8f4","black", "gray"]

# # TRANSFER FUNCTION
# ------------------------------------------

fig_dtk, axs_dtk = plt.subplots(ncols=1, nrows=1, figsize = (7,5))
for i in range(len(output_k)):
    k = output_k[i]
    axs_dtk.fill_between([0.5e-3,2.5e-2],-0.5,0.5,alpha = 0.05, color = "gray", edgecolor = None)
    axs_dtk.hlines(0,1e-7,1,ls = '--', colors = 'gray')
    # axs_dtk.plot((pert_nufld[k]['tau']-pert_ncdm[k]['tau'])/pert_ncdm[k]['tau'], pert_nufld[k]['delta_nufld[0]'], color = colors[i], lw = 3, label = r"$\mathrm{Modified}$"+r"$\, k = $ {}".format(k))
    # axs_dtk.plot(pert_nufld[k]['tau'],  pert_nufld[k]['delta_nufld[0]'], color = colors[i], ls = '-', lw = 2,  label = r"$\mathrm{Continuity}$"+r"$\, k = $ {}".format(k))
    axs_dtk.plot(pert_ncdm[k]['tau'],  pert_ncdm[k]['delta_ncdm[0]'], color = colors[i], ls = '-', lw = 2,  label = r"$\mathrm{Original}$"+r"$\, k = $ {}".format(k))
axs_dtk.set_xscale('log')
# axs_dtk.set_ylim([-0.5,0.5])
# axs_dtk.set_xlim([1e-3,1])
axs_dtk.set_xlim([1e3,2e4])
axs_dtk.set_xlabel(r'$\tau$')
axs_dtk.set_ylabel(r'$\delta(k)$')
axs_dtk.legend()
fig_dtk.tight_layout()
fig_dtk.savefig('delta_transfer.png')

# # SOUND SPEED
# -----------------------------------------
    
fig_cs2, axs_cs2 = plt.subplots(ncols=1, nrows=1, figsize = (7,5))
for i in range(3,len(output_k)):
    k = output_k[i]
    axs_cs2.fill_between([0.5e-3,2.5e-2],-0.05,0.5,alpha = 0.05, color = "gray", edgecolor = None)
    axs_cs2.hlines(0,1e-7,1,ls = '--', colors = 'gray')
    # axs_cs2.plot(pert_nufld[k]['a'], pert_nufld[k]['cs2_nufld[0]'], lw = 2,  ls = '-', color = colors[i], label = r"Modified, $k = $ "+str(k))
    axs_cs2.plot(pert_ncdm[k]['a'],  pert_ncdm[k]['cs2_fluid[0]'], lw = 1.5, ls = '-', color = colors[i], label = r"Original, $k = $ "+str(k))
    axs_cs2.set_xscale('log')
axs_cs2.set_ylim([-0.05,0.4])
axs_cs2.set_xlim([1e-6,1])
axs_cs2.legend(loc = "center left")
axs_cs2.set_xlabel(r'$a$')
axs_cs2.set_ylabel(r'$c_s^{\, 2}$')
fig_cs2.tight_layout()
fig_cs2.savefig('sound_speed.png')

# # SOUND SPEED COMPARISON
# -----------------------------------------

fig_cs2, axs_cs2 = plt.subplots(ncols=3, nrows=3, figsize = (15,15))
for i in range(len(output_k)):
    k = output_k[i]
    col = i%3
    row = int(i/3)
    # axs_cs2[row,col].fill_between([0.5e-3,2.5e-2],-0.05,0.5,alpha = 0.05, color = "gray", edgecolor = None)
    axs_cs2[row,col].hlines(0,1e-7,1,ls = '--', colors = 'gray')
    # axs_cs2.plot(pert_nufld[k]['a'], pert_nufld[k]['cs2_nufld[0]'], lw = 2,  ls = '-', color = colors[i], label = r"Modified, $k = $ "+str(k))
    axs_cs2[row,col].plot(pert_ncdm[k]['a'],  pert_ncdm[k]['cs2_fluid[0]'], lw = 3, ls = '-', color ='#3a5a40', label = r"$k = $ "+str(k), zorder = 10)
    axs_cs2[row,col].plot(pert_ncdm[k]['a'],  pert_ncdm[k]['cs2_gauge[0]'], lw = 1.5, ls = '--', color = '#a3b18a')
    axs_cs2[row,col].set_xscale('log')
    axs_cs2[row,col].set_ylim([-0.05,0.4])
    axs_cs2[row,col].set_xlim([1e-6,1])
    axs_cs2[row,col].legend(loc = "upper right", fontsize = 20)
    axs_cs2[row,col].set_xlabel(r'$a$')
    axs_cs2[row,col].set_ylabel(r'$c_s^{\, 2}$')
fig_cs2.suptitle(r'$\sum m_\nu =\ $'+mass+', l_max_ncdm = '+lmax)
fig_cs2.tight_layout()
fig_cs2.savefig('sound_speed_3x3_'+mass+'_'+lmax+'.png')


# # SOUND SPEED
# -----------------------------------------
    
# fig_cs2, axs_cs2 = plt.subplots(ncols=1, nrows=1, figsize = (7,5))
# for k in output_k:
#     axs_cs2.fill_between([0.5e-3,2.5e-2],-0.05,1.0,alpha = 0.05, color = "gray", edgecolor = None)
#     axs_cs2.hlines(0,1e-7,1,ls = '--', colors = 'gray')
#     # axs_cs2.scatter(pert_nufld[k]['a'], pert_nufld[k]['cs2_nufld[0]'], s = 1, label = r"Modified, $k = $ "+str(k))
#     axs_cs2.scatter(pert_ncdm[k]['a'],  pert_ncdm[k]['cs2_ncdm[0]'], s = 1, color = 'gray', alpha= 0.5, label = r"Original, $k = $ "+str(k))
# axs_cs2.set_xscale('log')
# axs_cs2.set_ylim([-0.05,1])
# axs_cs2.set_xlim([1e-4,1.1e-3])
# axs_cs2.legend(loc = 'upper left', fontsize = 12)
# axs_cs2.set_xlabel(r'$a$')
# axs_cs2.set_ylabel(r'$c_s^{\, 2}$')
# fig_cs2.tight_layout()
# fig_cs2.savefig('sound_speed_detailed.png')

# # SHEAR
# ----------------------------------------------

# fig_sig, axs_sig = plt.subplots(ncols=1, nrows=1, figsize = (7,5))
# k = output_k[0]
# axs_sig.hlines(0,1e-7,1,ls = '--', colors = 'gray')

# for i in range(len(output_k)):
#     k = output_k[i]
#     # axs_sig.plot(pert_nufld[k]['a'], pert_nufld[k]['shear_nufld[0]'], lw = 3, color = colors[i],label = "Our k = "+str(k))
#     axs_sig.plot(pert_ncdm[k]['a'], pert_ncdm[k]['shear_ncdm[0]'], lw = 2, color = colors[i], ls = '--', label = "CLASS k = "+str(k))

# axs_sig.set_xscale('log')
# axs_sig.legend()

# fig_dtk.tight_layout()
# fig_sig.savefig('shear.png')

exit()

# CONTINUITY EQUATIONS
# ----------------------------------------------------

fig_cont, axs_cont = plt.subplots(ncols=2, nrows = 1, figsize = (12,5))
ax_delta = axs_cont[0]
ax_theta = axs_cont[1]

def compute_delta(k,bkg_array, pert_array, species):
    w_bkg = bkg_array['p_'+species+'[0]']/bkg_array['rho_'+species+'[0]'] # This has a different shape :(
    w = np.interp(pert_array['a'],1/(1+bkg_array['z']),w_bkg) # This has been checked to work right!
    wprime = np.gradient(w,pert_array['tau'])

    a_prime_over_a = np.gradient(pert_array['a'],pert_array['tau'])/pert_array['a']

    delta = pert_array['delta_'+species+'[0]']
    theta = pert_array['theta_'+species+'[0]']
    shear = pert_array['shear_'+species+'[0]']

    phip = np.gradient(pert_array['phi'],pert_array['tau'])
    cs2 = pert_array['cs2_'+species+'[0]']

    deltarhs = -(1+w)*(theta-3*phip) -3*a_prime_over_a*(cs2-w)*delta
    deltalhs = np.gradient(pert_array['delta_'+species+'[0]'],pert_array['tau'])

    thetarhs = -a_prime_over_a*(1-3*w)*theta -wprime/(1+w)*theta + cs2/(1+w)*k**2*delta -k**2*shear +k**2*pert_array['psi']
    thetalhs = np.gradient(pert_array['theta_'+species+'[0]'],pert_array['tau'])

    return deltalhs, deltarhs, thetalhs, thetarhs

k = output_k[0]

deltalhs_nufld, deltarhs_nufld, thetalhs_nufld, thetarhs_nufld = compute_delta(k, bkg_nufld,pert_nufld[k],'nufld')
deltalhs_ncdm,  deltarhs_ncdm,  thetalhs_ncdm,  thetarhs_ncdm  = compute_delta(k, bkg_ncdm,pert_ncdm[k],'ncdm')
# deltalhs_ncdm  = np.gradient(pert_ncdm[k]['delta_ncdm[0]'],pert_ncdm[k]['tau'])

ax_delta.plot(pert_nufld[k]['a'],deltarhs_nufld, lw = 3, label = 'nufld RHS')
ax_delta.plot(pert_nufld[k]['a'],deltalhs_nufld, lw = 3, label = 'nufld LHS')
ax_delta.plot(pert_ncdm[k]['a'], deltarhs_ncdm, lw = 3, ls = '--', label = 'ncdm RHS')
ax_delta.plot(pert_ncdm[k]['a'], deltalhs_ncdm, lw = 3, ls = '--', label = 'ncdm LHS')
ax_theta.plot(pert_nufld[k]['a'],thetarhs_nufld, lw = 3, label = 'nufld RHS')
ax_theta.plot(pert_nufld[k]['a'],thetalhs_nufld, lw = 3, label = 'nufld LHS')
ax_theta.plot(pert_ncdm[k]['a'], thetarhs_ncdm, lw = 3, ls = '--', label = 'ncdm RHS')
ax_theta.plot(pert_ncdm[k]['a'], thetalhs_ncdm, lw = 3, ls = '--', label = 'ncdm LHS')
ax_delta.set_ylabel(r'$\dot\delta$')
ax_theta.set_ylabel(r'$\dot\theta$')
for ax in axs_cont:
    ax.set_xscale('log')
    ax.legend()
    ax.set_xlabel(r'$a$')
# ax_delta.set_ylim([-10,10])
# ax_theta.set_ylim([-10,10])
fig_cont.tight_layout()
fig_cont.savefig('continuity.png')

# # DIFFERENCE TO LAMBDA-CDM PLOTS
# # --------------------------------------------------

# fig_cl, axs_cl = plt.subplots(ncols = 1, nrows = 2, figsize = (6,9))
# for ax_cl in axs_cl:
#     ax_cl.hlines(0.0,2,2500, ls = '--', lw = 1, color = 'gray')
#     ax_cl.plot(pert_nufld['l'], pert_nufld['l']*(pert_nufld['l']+1)/(2*np.pi)*pert_nufld['TT'], color = 'c', ls = '--', lw = 2, label = r'Continuity equation')
#     ax_cl.plot(pert_ncdm['l'], pert_ncdm['l']*(pert_ncdm['l']+1)/(2*np.pi)*pert_ncdm['TT'], color = 'r', ls = '--', lw = 2, label = r'Full Boltzmann tower')
#     ax_cl.plot(pert_lcdm['l'], pert_lcdm['l']*(pert_lcdm['l']+1)/(2*np.pi)*pert_lcdm['TT'], color = 'k', ls = '--', lw = 1, label = r'LCDM')
#     # ax_cl.plot(pert_nofla['l'], (pert_nofla['TT']-pert_nofla['TT'])/pert_nofla['TT'], color = 'k', lw = 3, label = r'Exact')
#     # ax_cl.plot(pert_class['l'], (pert_class['TT']-pert_nofla['TT'])/pert_nofla['TT'], color = 'r', ls = '--', lw = 2, label = r'CLASS')
#     # ax_cl.plot(pert_caio['l'], (pert_caio['TT']-pert_nofla['TT'])/pert_nofla['TT'], color = 'c', ls = '--', lw = 2, label = r'Caio')
#     ax_cl.set_ylabel(r'$(C_\ell^\mathrm{TT}-C_{\ell,\mathrm{exact}}^\mathrm{TT})/C_{\ell,\mathrm{exact}}^\mathrm{TT}$')
#     ax_cl.set_ylabel(r'$\frac{l(l+1)}{2\pi}C_\ell^\mathrm{TT}$')
#     ax_cl.set_xlabel(r'$\ell$')
#     ax_cl.set_xlim(2,2500)
#     ax_cl.legend(fontsize = 12)

# axs_cl[0].set_xscale('log')
# axs_cl[1].set_xscale('linear')
# # ax_cl.set_xscale('log')

# # axs_cl[1].set_ylim(-0.2,0.2) # 1 eV
# # axs_cl[1].set_ylim(-0.03,0.03) # 0.1 eV
# # axs_cl[1].set_ylim(-0.002,0.002) # 0.01 eV

# fig_cl.tight_layout()
# fig_cl.savefig("comparison_cl_lensed_TT.png")

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

