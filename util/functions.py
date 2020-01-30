# project name: liine_hyde_so2
# created by diego aliaga daliaga_at_chacaltaya.edu.bo
import glob

import cartopy
import numpy
import pandas
import seaborn
import xarray
from cartopy import crs as ccrs
from matplotlib import pyplot
# from useful_scit.imps import *
from useful_scit.imps import (pd,np,xr,za,mpl,plt,sns, pjoin, os,glob,dt,sys,ucp,log, splot)

def sat_time_fix(sat_time):
    d0 = dt.datetime(1993, 1, 1)
    t0 = d0.timestamp()
    # t1 = float(x1.Time.median().values)
    d1 = dt.datetime.fromtimestamp(sat_time + t0)
    # dt1 = d1.date()
    return d1


def get_date_from_xr(xr):
    xa1 = xr
    tm = xa1.Time.median()
    dt1 = sat_time_fix(tm).date()
    return dt1


def plot_paper_so2(xa5):
    # %matplotlib inline
    # %config InlineBackend.figure_formats = ['svg']
    # plt.rc('text', usetex=False)
    # plt.rc('font', size=13)
    # plt.rc('font', sansserif='Arial')
    font = {'family': 'Arial',
            'weight': 'bold',
            #     'weight' : 'high',
            'size'  : 13,
            'style' : 'normal'}
    plt.rc('font', **font)
    plt.rcParams.update({
        'font.family'     : 'sans-serif',
        'font.sans-serif' : 'Arial',
        'font.style'      : 'normal',
        'xtick.labelsize' : 13,
        'ytick.labelsize' : 13,
        'axes.labelsize'  : 13,
        'mathtext.fontset': 'stixsans',
        'mathtext.default': 'regular',
        'text.usetex'     : False,
        #                 'text.latex.unicode' : True
    })
    plt.rcParams['axes.linewidth'] = 1.5
    cm = sns.blend_palette(sns.color_palette("coolwarm", 21)[10:], as_cmap=True)
    ax = plt.axes(projection=ccrs.Orthographic(0, 55))
    # xa5=xa4
    xa5.plot.contourf(
        ax=ax,
        transform=ccrs.PlateCarree(),
        x='lon_r', y='lat_r',
        vmin=0.0,
        vmax=20,
        cmap=cm,
        extend='neither',
        levels=11,
        cbar_kwargs={'shrink': 1}

    );
    ax.scatter(
        [24.289603],
        [61.844556],
        transform=ccrs.PlateCarree(),
        color='k'
    )
    # ax.set_global();
    ax.coastlines();
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax.gridlines()
    plt.rc('text', usetex=True)
    ax.figure.axes[1].set_ylabel(r'SO$_2$ [D.U.]');
    # ax.figure.set_figheight(5 / 2.54)
    ax.figure.set_figheight(7 / 2.54)
    # ax = ax
    ax.outline_patch.set_linewidth(1.5)
    return ax


def simple_plot_totals(ops, xa4):
    ax = xa4.plot(**ops)
    # plt.rc('text', usetex=True)
    # ax.figure.axes[0].set_xlabel(r'\textbf{longitude}')
    # ax.figure.axes[0].set_ylabel(r'\textbf{latitude}')
    # ax.figure.axes[1].set_ylabel(r'$\textbf{SO}_2$ [D.U.]');


def plot_total_divided_by_month(ops, xa3):
    xa3.tot.plot(col='dt', col_wrap=6, **ops)


def group_by_lat_lon(xa1):
    xa2 = xa1.groupby('lat_r').mean('lat_r')
    # %%
    xa3 = xa2.groupby('lon_r').mean('lon_r')

    return xa3


def assgign_lat_lon_coords(xa1):
    xa1 = xa1.assign_coords(lat_r=xa1.lat_r)
    xa1 = xa1.assign_coords(lon_r=xa1.lon_r)
    xa1 = xa1.swap_dims({'XDim_idx': 'lon_r'})
    xa1 = xa1.swap_dims({'YDim_idx': 'lat_r'})
    return xa1


def filter_quality_flags_below_zero(xa):
    good_flags_p = (xa.QualityFlags_PBL >= 0)
    xa['ColumnAmountSO2_PBL'] = xa['ColumnAmountSO2_PBL'].where(good_flags_p)
    good_flags_p = (xa.QualityFlags_STL >= 0)
    xa['ColumnAmountSO2_STL'] = xa['ColumnAmountSO2_STL'].where(good_flags_p)
    good_flags_p = (xa.QualityFlags_TRL >= 0)
    xa['ColumnAmountSO2_TRL'] = xa['ColumnAmountSO2_TRL'].where(good_flags_p)
    good_flags_p = (xa.QualityFlags_TRM >= 0)
    xa['ColumnAmountSO2_TRM'] = xa['ColumnAmountSO2_TRM'].where(good_flags_p)
    # %%
    xa1 = xa.mean(['sc_n'])
    return xa1


def import_omi_xarray(data_path, rounding_lat_lon=1 / 2):
    files = glob.glob(data_path + '*.he5')
    # %%
    ds_f = pd.DataFrame(files, columns=['path'])
    ds_f.sort_values('path', inplace=True)

    def open_path(p):
        xx = xr.open_dataset(
            p,
            group='HDFEOS/GRIDS/OMI Total Column Amount SO2/Data Fields')
        return xx

    ds_f['xa'] = ds_f.path.apply(lambda p: open_path(p))
    ds_f['dt'] = ds_f['xa'].apply(
        lambda p: pd.to_datetime(get_date_from_xr(p)))

    def set_time(r):
        x1 = r.xa.copy()
        x1 = x1.swap_dims({'phony_dim_1': 'YDim_idx'})
        x1 = x1.swap_dims({'phony_dim_2': 'XDim_idx'})

        x1['dt'] = r['dt']
        x1 = x1.expand_dims('dt')
        ll = len(x1.phony_dim_0)
        x1 = x1.assign_coords(sc_n=x1.phony_dim_0)
        x1 = x1.swap_dims({'phony_dim_0': 'sc_n'})

        #     ll= len(x1.phony_dim_1)
        #     x1 = x1.assign_coords(YDim_idx=x1.YDim_idx)

        #     ll= len(x1.phony_dim_2)
        #     x1 = x1.assign_coords(XDim_idx=x1.XDim_idx)
        #     x1=x1.swap_dims({'phony_dim_2':'XDim_idx'})

        return x1

    ds_f['xa'] = ds_f.apply(lambda r: set_time(r), axis=1)
    # ds_f
    # %%
    # %%
    xa = xr.concat(list(ds_f.xa.iloc[:]), dim='dt')
    # %%
    # %%
    lon = xa.Longitude.median(dim=['YDim_idx', 'dt', 'sc_n'])
    lat = xa.Latitude.median(dim=['XDim_idx', 'dt', 'sc_n'])
    # xa['lon']=lon
    # xa['lat']=lat
    # %%
    xa = xa.assign_coords(lat=lat)
    xa = xa.assign_coords(lon=lon)
    # %%
    xa['lat_r'] = (rounding_lat_lon * xa.lat).round() / rounding_lat_lon
    xa['lon_r'] = (rounding_lat_lon * xa.lon).round() / rounding_lat_lon
    return xa


def sum_over_all_layers(xa3):
    xa3['tot'] = (
            xa3.ColumnAmountSO2_PBL + \
            xa3.ColumnAmountSO2_STL + \
            xa3.ColumnAmountSO2_TRL + \
            xa3.ColumnAmountSO2_TRM

    )

    return xa3


def get_hysplit_xarray(files, lat_lon_rounding=2, power_distance_norm=2):
    '''
    gets and process hysplit files.
        - processing includes:
            - divide results by one millon? normalized?
            - group by lat_lon_rounding (2 degrees)
    :param files:
    :param lat_lon_rounding:
    :return: xr.xarray
    '''
    dff = pd.DataFrame(files, columns=['path'])
    # dff
    # %%
    f1 = files[10]
    # %%
    cols = ['i', 'one', 'y', 'm', 'd', 'h', 'u', 'u0', 'hb', 'lat', 'lon', 'z',
            '12', '13', '14', '15', '16', '17', '18', '19']

    def get_hys(path):
        df1 = pd.read_csv(path, sep='\s+', skiprows=19, header=None, names=cols)
        df1 = df1[(df1.i >= 4) & (df1.i <= 6)]
        return df1

    # %%
    dff['df'] = dff.path.apply(lambda p: get_hys(p))
    # %%
    ndf = pd.concat(dff.iloc[:].df.values)
    # %%
    lat1, lon1 = 61.850, 24.290

    def dis(la1, lo1, la2, lo2):
        """calculate distance in la lo coordinate. prob not super accurate."""
        ret = np.sqrt(
            (la1 - la2) ** 2 + (lo1 - lo2) ** 2
        )
        return ret

    # dis(0, 0, 1, 1)
    # %%
    dis_from_hyyde = 'dis'
    ndf[dis_from_hyyde] = ndf.apply(lambda r: dis(lat1, lon1, r.lat, r.lon), axis=1)
    # %%
    ndff = ndf.copy()
    # %%
    # 47, 81, -40, 40
    ndff = ndf.copy()
    ndff['lat_r'] = np.round(ndff.lat / lat_lon_rounding) * lat_lon_rounding
    ndff['lon_r'] = np.round(ndff.lon / lat_lon_rounding) * lat_lon_rounding
    ndff = ndff[
        (ndff.lat >= 50) &
        (ndff.lat <= 80) &
        (ndff.lon >= -34) &
        (ndff.lon <= 38)
        ]
    ndf1 = ndff.groupby(['lat_r', 'lon_r']).count()['i']
    ndf2 = ndff.groupby(['lat_r', 'lon_r']).mean()[dis_from_hyyde]
    # %%
    nndf = (ndf2 ** power_distance_norm) * ndf1
    # %%
    # xa1 = ndf1.to_xarray()
    xa1 = nndf.to_xarray()
    xa1 = xa1.where(xa1 >= 0, 0) / 10 ** 6
    return xa1


def plot_paper_hysplit(xa1):
    plt.rc('text', usetex=False)
    # plt.rc('font', size=13)
    # plt.rc('font', sansserif='Arial')
    font = {'family': 'Arial',
            'weight': 'bold',
            #     'weight' : 'high',
            'size'  : 13,
            'style' : 'normal'}
    plt.rc('font', **font)
    # %%
    # xa1['i'].plot()
    # %config InlineBackend.figure_formats = ['svg']
    ax = plt.axes(projection=ccrs.Orthographic(0, 55))
    plt.rcParams.update({
        'font.family'     : 'sans-serif',
        'font.sans-serif' : 'Arial',
        'font.style'      : 'normal',
        'xtick.labelsize' : 13,
        'ytick.labelsize' : 13,
        'axes.labelsize'  : 13,
        'mathtext.fontset': 'stixsans',
        'mathtext.default': 'regular',
        'text.usetex'     : False,
        #                 'text.latex.unicode' : True
    })
    cm = sns.blend_palette(sns.color_palette("coolwarm", 21)[10:], as_cmap=True)
    plt.rcParams['axes.linewidth'] = 1.5
    xa1.plot(
        x='lon_r',
        y='lat_r',
        #     kind='scatter',
        ax=ax,
        transform=ccrs.PlateCarree(),
        vmin=0.0,
        vmax=1.0,
        cmap=cm,
        extend='neither',
        levels=11,
        cbar_kwargs={'shrink': 1},
    )
    ax.scatter(
        [24.289603],
        [61.844556],
        transform=ccrs.PlateCarree(),
        color='k'
    )
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax.gridlines()
    # ax.set_global()
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim());
    ax.figure.set_figheight(5 / 2.54)
    ax.figure.set_figheight(7 / 2.54)
    axb = ax.figure.axes[1]
    axb.set_ylabel('Normalized Freq.')
    ax.outline_patch.set_linewidth(1.5)
    return ax