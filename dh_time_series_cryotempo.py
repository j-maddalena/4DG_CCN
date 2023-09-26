#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:12:50 2023

@author: maddalen
"""
#load python modules
import numpy as np
import matplotlib.pyplot as plt
#import pyproj
from scipy.ndimage import gaussian_filter

#load cpom modules
from cpom.areas.area_plot import Polarplot, Annotation
from cpom.gridding.gridareas import GridArea 
from cpom.areas.areas import Area
from cpom.masks.masks import Mask

# Load functions 
def dh_mean(basin_name, dh, time_vector, lats, lons):
    '''
    Calculates the mean dh for a given area basin for each epoch and associated errors
    Based on Tom Slater's dz_mean.mat 
    Currently set up to mask on the 4DG basins but need to add ablation zone masking.

    Parameters
    ----------
    basin_name : string
        name of 4DG basin to load as an area using area.py either 'watson' or 'negis'.
    dh : 3D array
        stacked grids of elevation change dh outputted from epoch_average.py from the cpom software (timestep, row, column).
    time_vector : array
        currently the number of time steps from the epoch averaging but will need to update to times.
    lats : array
        the corresponding latitude for the centre of the dh grid cells.
    lons : array
        the corresponding longitude for the centre of the dh grid cells..

    Returns
    -------
    dh_basin_mean : array
        mean of dh at each epoch.
    dh_basin_sigma : array
        standard deviation of dh / sqrt(number of dh measurement in basin).
    dh_basin_sigma_cum : array
        cumlatively adding dh_basin_sigma.
    dh_basin_mean_smooth : array
        gaussian 3 sigma smoother applied to mean of dh at each epoch.

    '''
    
    
    thisarea = Area(basin_name) # load the 4DG basin area
    basin_mask = Mask(thisarea.maskname) # loads basin mask
    
    inmask,_,_ = basin_mask.points_inside(lats,lons) #creates 1D array boelan mask for basin

       
    inmask_reshaped = inmask.reshape(dh.shape[1], dh.shape[2]) # Reshape the inmask array to match the dimensions of dh
    dh_masked = np.where(inmask_reshaped, dh, np.nan) # mask out unwanted values in dh

    '''
    #show example of data
    dh_masked_example = dh_masked[0, :, :]
    dh_masked_example = dh_masked_example.flatten()
    Polarplot('negis').plot_points(
        lats=lats,
        lons=lons,
        vals=dh_masked_example,
        varname="dh",
        varunits=" ",
        title="Test plot: dh first epoch",
        plotrange=[-1,1],
        cmap="RdBu",
        #background=None,
        # cmap_under_color='#A85754',
        # cmap_over_color='#3E4371',Polarplot
        # cmap_extend='both',
        scattersize=60000,
        # annotation_list=annotation_list,
        #draw_fillvalue_map=False
        outdir=plot_outdir
    )
    '''

    # calculate mean for each time step
    dh_basin_mean = np.full(np.shape(dh)[0],np.nan)
    dh_basin_sigma = np.full(np.shape(dh)[0],np.nan)
    # Mean for basin and error at each time step
    for i in range(np.shape(dh)[0]):
        tmp = dh_masked[i, :, :]
        dh_basin_mean[i] = np.nanmean(tmp)
        n_cells = np.sum(~np.isnan(tmp))
        dh_basin_sigma[i] = np.nanstd(tmp) / np.sqrt(n_cells)
    
    
    dh_basin_sigma_cum = np.empty_like(dh_basin_sigma) * np.nan
    for i in range(np.shape(dh)[0]):
        dh_basin_sigma_cum[i] = np.sqrt(np.sum(dh_basin_sigma[:i+1]**2)) # Accumulate error in time
        
    dh_basin_mean_smooth = gaussian_filter(dh_basin_mean, sigma=3) # Smooth with gaussain to preserve seasonality
    
    return dh_basin_mean, dh_basin_sigma, dh_basin_sigma_cum, dh_basin_mean_smooth

#usefel directories
plot_outdir = '/media/luna/maddalen/4DG/Example plots/'
grid_dir = '/media/luna/maddalen/4DG/CryoTEMPO/sec_runs/toms_method/epoch_averaged/30_day/'

#load data from epoch average npz file greenland
cryotempo_epoch_ave_30day = np.load(grid_dir + 'epoch_average_cs2_greenland.npz',allow_pickle=True)
#cryotempo_epoch_ave_30day.files
dh = cryotempo_epoch_ave_30day['dh_ave'][:] # dh time series 
dh_start_time = cryotempo_epoch_ave_30day['input_dh_start_time'][:] 
dh_end_time = cryotempo_epoch_ave_30day['input_dh_end_time'][:] 
#dh_stddev = cryotempo_epoch_ave_30day['input_dh_stddev'] # useful?

dh_time_midpoint_of_epoch =  (dh_start_time + dh_end_time)/2
epoch_year = 1991
time_mid_point_epoch_in_decimal_years = dh_time_midpoint_of_epoch + epoch_year # get time in decimal years
del dh_start_time, dh_end_time, dh_time_midpoint_of_epoch, epoch_year

#get lat, lon and x,y from gris
thisgridarea = GridArea('greenland', 5000)
ncols = np.shape(dh)[2]
nrows = np.shape(dh)[1]
x = []
y = []
lats = []
lons = []
for col in range(ncols):
    for row in range(nrows):
        thisx, thisy = thisgridarea.get_cellcentre_x_y_from_col_row(col, row)
        x.append(thisx)
        y.append(thisy)
if len(x) > 0:
    # Much quicker to do this once for an array of x,y than 1 by 1
    lats, lons = thisgridarea.transform_x_y_to_lat_lon(x, y)
del col, row, ncols, nrows, thisx, thisy
del grid_dir, thisgridarea


#test one grid cell near summit
dh_303_322 = dh[:,322,303]
#start_time_303_322 = dh_start_time[:,322,303]
epochs = np.arange(1, np.shape(dh)[0]+1)
plt.plot(epochs,dh_303_322)
#del dh_303_322,start_time_303_322


dh_watson_mean, dh_watson_sigma, dh_watson_sigma_cum, dh_watson_mean_smooth = dh_mean('watson', dh, epochs, lats, lons)
dh_negis_mean, dh_negis_sigma, dh_negis_sigma_cum, dh_negis_mean_smooth = dh_mean('negis', dh, epochs, lats, lons)


#watson plots 
plt.errorbar(epochs, dh_watson_mean, yerr=dh_watson_sigma, fmt='.', label='dh with sigma')
plt.title('Test plot: mean dh over Watson per epoch raw data without smoothing etc \nerror bars are std/sqrt(n_cells)')
plt.xlabel('epoch')
plt.ylabel('dh (m)')
#plt.savefig(plot_outdir + 'watson_dh_raw.png')

plt.errorbar(epochs, dh_watson_mean, yerr=dh_watson_sigma_cum, fmt='.', label='dh with sigma')
plt.title('Test plot: mean dh over Watson per epoch, raw data without smoothing etc \nwith cumulative error ')
plt.xlabel('epoch')
plt.ylabel('dh (m)')
#plt.savefig(plot_outdir + 'watson_dh_raw_cumulative_error.png')

plt.errorbar(epochs, dh_watson_mean_smooth, yerr=dh_watson_sigma, fmt='.', label='dh with sigma')
plt.title('Test plot: mean dh over Watson per epoch with 3sigma gaussian smoothing \nerror bars are std/sqrt(n_cells)')
plt.xlabel('epoch')
plt.ylabel('dh (m)')
#plt.savefig(plot_outdir + 'watson_dh_smooth.png')

#negis plots 
plt.errorbar(epochs, dh_negis_mean, yerr=dh_negis_sigma, fmt='.', label='dh with sigma')
plt.title('Test plot: mean dh over NEGIS per epoch raw data without smoothing etc \nerror bars are std/sqrt(n_cells)')
plt.xlabel('epoch')
plt.ylabel('dh (m)')
#plt.savefig(plot_outdir + 'negis_dh_raw.png')

plt.errorbar(epochs, dh_negis_mean, yerr=dh_negis_sigma_cum, fmt='.', label='dh with sigma')
plt.title('Test plot: mean dh over NEGIS per epoch, raw data without smoothing etc \nwith cumulative error ')
plt.xlabel('epoch')
plt.ylabel('dh (m)')
#plt.savefig(plot_outdir + 'negis_dh_raw_cumulative_error.png')

plt.errorbar(epochs, dh_negis_mean_smooth, yerr=dh_negis_sigma, fmt='.', label='dh with sigma')
plt.title('Test plot: mean dh over NEGIS per epoch raw data with 3sigma gaussian smoothing \nerror bars are std/sqrt(n_cells)')
plt.xlabel('epoch')
plt.ylabel('dh (m)')
#plt.savefig(plot_outdir + 'negis_dh_smooth.png')




