# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # import stuff
# %%
from useful_scit.imps import (pd,np,xr,za,mpl,plt,sns, pjoin, os,glob,dt,sys,ucp,log, splot)

sys.path.insert(0, "../")

import util.functions as fun

# import pynio

# %% [markdown]
# # OMI SO2
# %%
# def main():
# global open_path, set_time, cols, get_hys, lat1, lon1, dis
data_path = '../data/omi/'
xa = fun.import_omi_xarray(data_path)
# %% {"jupyter": {"outputs_hidden": true}}
xa1 = fun.filter_quality_flags_below_zero(xa)
# %%
xa1 = fun.assgign_lat_lon_coords(xa1)
# %% {"jupyter": {"outputs_hidden": true}}
xa3 = fun.group_by_lat_lon(xa1)
xa3 = fun.sum_over_all_layers(xa3)
xa4 = xa3['tot'].mean('dt')
xa5 = xa4.where(xa4 >= 0, 0)
# %% {"jupyter": {"outputs_hidden": true}}
ops = dict(x='lon_r', y='lat_r', vmin=0, vmax=20)
fun.plot_total_divided_by_month(ops, xa3)
# %% {"jupyter": {"outputs_hidden": true}}
fun.simple_plot_totals(ops, xa4)
# %%
ax = fun.plot_paper_so2(xa5)
ax.figure.savefig('./so2.png', dpi=300)
ax.figure.savefig('./so2.pdf')
# %% [markdown]
# # hysplit stuff
# %%
files = glob.glob('../data/hysplit/hysplit201409*.traj')
xa1 = fun.get_hysplit_xarray(files,power_distance_norm=1)
# %%
ax = fun.plot_paper_hysplit(xa1/xa1.max())
ax.figure.savefig('./hys.png', dpi=300)
ax.figure.savefig('./hys.pdf')
# %%

# %%
# main()

# %%

# %%
