# Get an instance of data (hour over lake)
import os
import csv
import time
import gzip
import glob
import shutil
import random
import logging
import requests
import pywgrib2_s
import multiprocessing
import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage as nd
import metpy.calc as mpcalc
from cdo import *
from metpy.units import units
from herbie import FastHerbie
from datetime import datetime, timedelta



def create_dir(folder_name):
    backup_folder_path = os.path.join('dataset/original', folder_name)
    os.makedirs(backup_folder_path, exist_ok=True)
    subfolders = ['hrrr', 'mrms']
    for subfolder in subfolders:
        os.makedirs(os.path.join(backup_folder_path, subfolder), exist_ok=True)


def gaussian_filter_2d(u, sigma):
    return xr.apply_ufunc(
        lambda x: nd.gaussian_filter(x, sigma=[sigma, sigma]), u,
        input_core_dims=[['lat', 'lon']],
        output_core_dims=[['lat', 'lon']],
        vectorize=True
    )


def mfilerdir_hrrr(directory):
    items = os.listdir(directory)
    for item in items:
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    new_file_name = file.split("__", 1)[-1]
                    original_folder_name = os.path.basename(item_path)
                    new_path = os.path.join(directory, original_folder_name + "_" + new_file_name)
                    shutil.move(file_path, new_path)
            shutil.rmtree(item_path)


def get_hrrr(dirname, htime, thds, lake):
    DATES = pd.date_range(start=(htime - timedelta(hours=1)).strftime("%Y-%m-%d %H:00"), periods=1, freq="1H")
    data = FastHerbie(DATES, model="hrrr", product="prs", fxx=range(1,2), max_threads=thds)
    data.download(
        searchString="((TMP|DPT|UGRD|VGRD):(850|925))|((UGRD|VGRD):10 m)|((TMP|PRES|CAPE|ICEC):surface)|(DPT:2 m)",
        max_threads=thds,
        save_dir=f"./dataset/original/{dirname}/"
    )
    
    mfilerdir_hrrr(f"./dataset/original/{dirname}/hrrr/")
    file = glob.glob(f"./dataset/original/{dirname}/hrrr/*.grib2")[0]
    pywgrib2_s.wgrib2([file, "-netcdf", file.replace(".grib2", "_1.nc")])
    file = file.replace(".grib2", "_1.nc")
    cdo.remapnn(f"./grids/{lake}", input=file, options="-f nc4", output=file.replace("_1.nc", "_2.nc"))
    file = file.replace("_1.nc", "_2.nc")
    cdo.merge(input=f"{file} ./dem/dem_{lake}.nc", options=f"-b F32 -P {thds} -f nc -r", output=f"./dataset/original/{dirname}/hrrr.nc")


def get_mrms_iowa(dirname, htime, thds, lake):
    DATES = pd.date_range(start=htime.strftime("%Y-%m-%d %H:00"), periods=2, freq="1H")
    base = "https://mtarchive.geol.iastate.edu/"
    ext1 = "/mrms/ncep/GaugeCorr_QPE_01H/GaugeCorr_QPE_01H"
    ext2 = "/mrms/ncep/SeamlessHSR/SeamlessHSR"
    
    # -1 to 0 hour pcp
    url = f"{base}{DATES[0].strftime('%Y/%m/%d')}{ext1}_00.00_{DATES[0].strftime('%Y%m%d')}-{DATES[0].strftime('%H0000')}.grib2.gz"
    response = requests.get(url, stream=True)
    grib_file = f"./dataset/original/{dirname}/mrms/QPE_past.grib2.gz"
    with open(grib_file, 'wb') as f:
        f.write(response.content)
    
    # SHSR
    url = f"{base}{DATES[0].strftime('%Y/%m/%d')}{ext2}_00.00_{DATES[0].strftime('%Y%m%d')}-{DATES[0].strftime('%H0000')}.grib2.gz"
    response = requests.get(url, stream=True)
    grib_file = f"./dataset/original/{dirname}/mrms/SHSR_mrms.grib2.gz"
    with open(grib_file, 'wb') as f:
        f.write(response.content)
    
    # 0 to 1 hour pcp
    url = f"{base}{DATES[1].strftime('%Y/%m/%d')}{ext1}_00.00_{DATES[1].strftime('%Y%m%d')}-{DATES[1].strftime('%H0000')}.grib2.gz"
    response = requests.get(url, stream=True)
    grib_file = f"./dataset/original/{dirname}/mrms/QPE_target.grib2.gz"
    with open(grib_file, 'wb') as f:
        f.write(response.content)
    
    grib_files = glob.glob(f"./dataset/original/{dirname}/mrms/*.grib2.gz")
    for file in grib_files:
        with gzip.open(file, 'rb') as f_in:
            with open(file.replace('.gz', ''), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    grib_files = glob.glob(f"./dataset/original/{dirname}/mrms/*.grib2")
    for file in grib_files:
        pywgrib2_s.wgrib2([file, "-netcdf", file.replace(".grib2", ".nc")])
        file = file.replace(".grib2", ".nc")
        cdo.remapnn(f"./grids/{lake}", input=file, options="-f nc4", output=file.replace(".nc", "_1.nc"))
        fname =  os.path.basename(file).replace(".nc", "")
        if fname == 'QPE_past':
            cdo.settaxis(f"{DATES[0].strftime('%Y-%m-%d,%H:%M:%S,1hour')}", input=f"-chname,GaugeCorrQPE01H_0mabovemeansealevel,{fname} {file.replace('.nc', '_1.nc')}", options="-f nc4 -r", output=file.replace(".nc", "_3.nc"))
        elif fname == 'QPE_target':
            cdo.settaxis(f"{DATES[0].strftime('%Y-%m-%d,%H:%M:%S,1hour')}", input=f"-chname,GaugeCorrQPE01H_0mabovemeansealevel,{fname} {file.replace('.nc', '_1.nc')}", options="-f nc4 -r", output=file.replace(".nc", "_3.nc"))
        else:
            cdo.settaxis(f"{DATES[0].strftime('%Y-%m-%d,%H:%M:%S,1hour')}", input=f"-chname,SeamlessHSR_0mabovemeansealevel,{fname} {file.replace('.nc', '_1.nc')}", options="-f nc4 -r", output=file.replace(".nc", "_2.nc"))
            cdo.setmisstoc("0", input=f"-setrtomiss,-1000,0 {file.replace('.nc', '_2.nc')}", options="-f nc4", output=file.replace(".nc", "_3.nc"))

    nc_files = [file.replace(".grib2", "_3.nc") for file in grib_files]
    cdo.merge(input=f"{' '.join(nc_files)}", options=f"-b F32 -P {thds} -f nc -r", output=f"./dataset/original/{dirname}/mrms.nc")


def get_mrms_aws(dirname, htime, thds, lake):
    DATES = pd.date_range(start=htime.strftime("%Y-%m-%d %H:00"), periods=2, freq="1H")
    base = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/"
    ext1 = "MultiSensor_QPE_01H_Pass1_00.00"
    ext2 = "SeamlessHSR_00.00"
    ext3 = "MultiSensor_QPE_01H_Pass2_00.00"
    
    # -1 to 0 hour pcp
    url = f"{base}{ext1}/{DATES[0].strftime('%Y%m%d')}/MRMS_{ext1}_{DATES[0].strftime('%Y%m%d-%H0000')}.grib2.gz"
    response = requests.get(url, stream=True)
    grib_file = f"./dataset/original/{dirname}/mrms/QPE_past.grib2.gz"
    with open(grib_file, 'wb') as f:
        f.write(response.content)
    
    # SHSR
    url = f"{base}{ext2}/{DATES[0].strftime('%Y%m%d')}/MRMS_{ext2}_{DATES[0].strftime('%Y%m%d-%H0000')}.grib2.gz"
    response = requests.get(url, stream=True)
    grib_file = f"./dataset/original/{dirname}/mrms/SHSR_mrms.grib2.gz"
    with open(grib_file, 'wb') as f:
        f.write(response.content)
    
    # 0 to 1 hour pcp
    url = f"{base}{ext3}/{DATES[1].strftime('%Y%m%d')}/MRMS_{ext3}_{DATES[1].strftime('%Y%m%d-%H0000')}.grib2.gz"
    response = requests.get(url, stream=True)
    grib_file = f"./dataset/original/{dirname}/mrms/QPE_target.grib2.gz"
    with open(grib_file, 'wb') as f:
        f.write(response.content)
    
    grib_files = glob.glob(f"./dataset/original/{dirname}/mrms/*.grib2.gz")
    for file in grib_files:
        with gzip.open(file, 'rb') as f_in:
            with open(file.replace('.gz', ''), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    grib_files = glob.glob(f"./dataset/original/{dirname}/mrms/*.grib2")
    for file in grib_files:
        pywgrib2_s.wgrib2([file, "-netcdf", file.replace(".grib2", ".nc")])
        file = file.replace(".grib2", ".nc")
        cdo.remapnn(f"./grids/{lake}", input=file, options="-f nc4", output=file.replace(".nc", "_1.nc"))
        fname =  os.path.basename(file).replace(".nc", "")
        if fname == 'QPE_past':
            cdo.settaxis(f"{DATES[0].strftime('%Y-%m-%d,%H:%M:%S,1hour')}", input=f"-chname,var209_6_30_0mabovemeansealevel,{fname} {file.replace('.nc', '_1.nc')}", options="-f nc4 -r", output=file.replace(".nc", "_3.nc"))
        elif fname == 'QPE_target':
            cdo.settaxis(f"{DATES[0].strftime('%Y-%m-%d,%H:%M:%S,1hour')}", input=f"-chname,var209_6_37_0mabovemeansealevel,{fname} {file.replace('.nc', '_1.nc')}", options="-f nc4 -r", output=file.replace(".nc", "_3.nc"))
        else:
            cdo.settaxis(f"{DATES[0].strftime('%Y-%m-%d,%H:%M:%S,1hour')}", input=f"-chname,SeamlessHSR_0mabovemeansealevel,{fname} {file.replace('.nc', '_1.nc')}", options="-f nc4 -r", output=file.replace(".nc", "_2.nc"))
            cdo.setmisstoc("0", input=f"-setrtomiss,-1000,0 {file.replace('.nc', '_2.nc')}", options="-f nc4", output=file.replace(".nc", "_3.nc"))

    nc_files = [file.replace(".grib2", "_3.nc") for file in grib_files]
    cdo.merge(input=f"{' '.join(nc_files)}", options=f"-b F32 -P {thds} -f nc -r", output=f"./dataset/original/{dirname}/mrms.nc")


def get_mrms(dirname, htime, thds, lake):
    if htime < datetime(2020, 10, 15): get_mrms_iowa(dirname, htime, thds, lake)
    else: get_mrms_aws(dirname, htime, thds, lake)


def merge(dirname, lake, save):
    ds1 = xr.open_dataset(f"./dataset/original/{dirname}/hrrr.nc")
    ds1 = ds1.isel(time=0, drop=True)
    ds1 = ds1.rename({'DPT_2maboveground': 'DPT_2m', 'UGRD_10maboveground': 'UGRD_10m', 'VGRD_10maboveground': 'VGRD_10m'})
    ds1['TMP_masked'] = ds1['TMP_surface'] * ds1['landsea']
    ds1['TMP_masked'].attrs['units'] = 'K'
    
    ds1['slope'] = np.deg2rad(ds1['slope'])
    ds1['aspect'] = np.deg2rad(ds1['aspect'])
    ds1['flow'] = -np.tan(ds1['slope']) * (ds1['UGRD_10m'] * np.sin(ds1['aspect']) + ds1['VGRD_10m'] * np.cos(ds1['aspect']))
    ds1['flow'] = gaussian_filter_2d(ds1['flow'], 2)
    
    offset = xr.where(ds1['ICEC_surface'] == 0, 0.1, 0.5)
    ds1['DPT_surface'] = xr.where(ds1['ICEC_surface'] == 0, ds1['TMP_surface']-offset, ds1['DPT_2m'])
    ds1['DPT_surface'] = xr.where(ds1['DPT_surface'] > ds1['TMP_surface']-offset, ds1['TMP_surface']-offset, ds1['DPT_surface']) * units.kelvin
    ds1['THTE_masked'] = mpcalc.equivalent_potential_temperature(ds1['PRES_surface'], ds1['TMP_surface'], ds1['DPT_surface']).metpy.dequantify() * ds1['landsea']
    ds1['THTE_850mb'] = mpcalc.equivalent_potential_temperature(850*units.mbar, ds1['TMP_850mb'], ds1['DPT_850mb']).metpy.dequantify()
    
    u_s = gaussian_filter_2d(ds1['UGRD_925mb'].metpy.quantify(), 2)
    v_s = gaussian_filter_2d(ds1['VGRD_925mb'].metpy.quantify(), 2)
    u = (u_s * units.meters / units.seconds).metpy.quantify()
    v = (v_s * units.meters / units.seconds).metpy.quantify()
    dx, dy = mpcalc.lat_lon_grid_deltas(ds1['lon'].values * units.degrees_east, ds1['lat'].values * units.degrees_north)
    ds1['RELV_925mb'] = mpcalc.vorticity(u, v, dx=dx, dy=dy).metpy.dequantify()
    ds1['DIVG_925mb']  = mpcalc.divergence(u, v, dx=dx, dy=dy).metpy.dequantify()
    
    ds1 = ds1.drop_vars(['slope', 'aspect', 'DPT_surface', 'UGRD_10m', 'VGRD_10m', 'PRES_surface'])
    ds2 = xr.open_dataset(f"./dataset/original/{dirname}/mrms.nc")
    ds2 = ds2.isel(time=0, drop=True)
    vars = [var for var in ds2.data_vars if var not in ds2.dims]
    for var in vars:
        if var == 'SHSR_mrms': ds2[var].attrs = {'units': 'dBz'}
        else: ds2[var].attrs = {'units': 'mm'}
    ds = xr.merge([ds1, ds2])
    ds1.close()
    ds2.close()
    
    for var in ds.data_vars: ds[var].attrs = {}
    ds = ds.rename({'lat': 'y', 'lon': 'x'})
    ds = ds.assign_coords(y=("y", range(len(ds.coords['y']))))
    ds = ds.assign_coords(x=("x", range(len(ds.coords['x']))))
    ds = ds.transpose('y', 'x')
    ds = xr.Dataset(
        {var: (['y', 'x'], ds[var].values) for var in ds.data_vars},
        coords={
            'y': ds.coords['y'],
            'x': ds.coords['x'],
        }
    )
    
    ins = ds.drop_vars("QPE_target")
    out = ds["QPE_target"]
    if lake == 'm':
        ins = ins.chunk({'y': 512, 'x': 256})
        out = out.chunk({'y': 512, 'x': 256})
    else:
        ins = ins.chunk({'y': 256, 'x': 512})
        out = out.chunk({'y': 256, 'x': 512})
    ins.to_netcdf(f"./dataset/{dirname}_input.nc", format="NETCDF3_64BIT") # ins.to_zarr(f"./dataset/{save}/{dirname}_input.zarr", mode='w', consolidated=True)
    out.to_netcdf(f"./dataset/{dirname}_target.nc", format="NETCDF3_64BIT") # out.to_zarr(f"./dataset/{save}/{dirname}_target.zarr", mode='w', consolidated=True)
    ds.close()


def process_day(date, thds, lake, save):
    dirname = f"{date.strftime('%Y%m%d_%H')}{lake}"
    create_dir(dirname)
    print("\n\n Starting MRMS \n\n")
    get_mrms(dirname, date, thds//2, lake)
    print("\n\n DONE with MRMS ")
    print("\n Starting HRRR \n\n")
    get_hrrr(dirname, date, thds//2, lake)
    print("\n\n DONE with HRRR ")
    print("\n Starting merge \n\n")
    merge(dirname, lake, save)
    print("\n\n DONE with merge \n\n")
    shutil.rmtree(f"./dataset/original/{dirname}/")


def main(thds=32):
    st = time.time()
    os.environ['REMAP_EXTRAPOLATE'] = 'off'
    
    get_date = datetime(2021, 1, 18, 0, 0)
    process_day(get_date, thds, 'm', 'train')
    
    et = time.time()
    ti = et - st
    print(f'\nDone in {ti} seconds')


if __name__ == "__main__":
    cdo = Cdo()
    main()
