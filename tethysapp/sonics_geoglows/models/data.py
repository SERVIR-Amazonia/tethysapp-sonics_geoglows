####################################################################################################
##                                   LIBRARIES AND DEPENDENCIES                                   ##
####################################################################################################

# Geoglows
import geoglows
import numpy as np
import math
import hydrostats as hs
import hydrostats.data as hd
import HydroErr as he
import plotly.graph_objs as go
import datetime as dt
import pandas as pd

# Sonics
from glob import glob
import os
import xarray as xr
from lmoments3 import distr




####################################################################################################
##                                 UTILS AND AUXILIAR FUNCTIONS                                   ##
####################################################################################################

def get_format_data(sql_statement, conn):
    # Retrieve data from database
    data =  pd.read_sql(sql_statement, conn)
    # Datetime column as dataframe index
    data.index = data.datetime
    data = data.drop(columns=['datetime'])
    # Format the index values
    data.index = pd.to_datetime(data.index)
    data.index = data.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
    data.index = pd.to_datetime(data.index)
    data[data < 0.04] = 0.04
    # Return result
    return(data)


def get_bias_corrected_data(sim, obs):
    outdf = geoglows.bias.correct_historical(sim, obs)
    outdf.index = pd.to_datetime(outdf.index)
    outdf.index = outdf.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
    outdf.index = pd.to_datetime(outdf.index)
    return(outdf)


def gve_1(loc: float, scale: float, shape: float, rp: int or float) -> float:
    gve = ((scale / shape) * (1 - math.exp(shape * (math.log(-math.log(1 - (1 / rp))))))) + loc
    return(gve)


def get_return_periods(comid, data_df):
    max_annual_flow = data_df.groupby(data_df.index.strftime("%Y")).max()
    params = distr.gev.lmom_fit(max_annual_flow.iloc[:, 0].values.tolist())
    return_periods_values_g = []
    #
    return_periods = [10, 5, 2.33]
    for rp in return_periods:
        return_periods_values_g.append(gve_1(params['loc'], params['scale'], params['c'], rp))
    #
    d = {'rivid': [comid], 
         'return_period_10': [return_periods_values_g[0]],
         'return_period_5': [return_periods_values_g[1]],
         'return_period_2_33': [return_periods_values_g[2]]}
    #
    rperiods = pd.DataFrame(data=d)
    rperiods.set_index('rivid', inplace=True)
    return(rperiods)


def ensemble_quantile(ensemble, quantile, label):
    df = ensemble.quantile(quantile, axis=1).to_frame()
    df.rename(columns = {quantile: label}, inplace = True)
    return(df)


def get_ensemble_stats(ensemble):
    high_res_df = ensemble['ensemble_52_m^3/s'].to_frame()
    ensemble.drop(columns=['ensemble_52_m^3/s'], inplace=True)
    ensemble.dropna(inplace= True)
    high_res_df.dropna(inplace= True)
    high_res_df.rename(columns = {'ensemble_52_m^3/s':'high_res_m^3/s'}, inplace = True)
    stats_df = pd.concat([
        ensemble_quantile(ensemble, 1.00, 'flow_max_m^3/s'),
        ensemble_quantile(ensemble, 0.75, 'flow_75%_m^3/s'),
        ensemble_quantile(ensemble, 0.50, 'flow_avg_m^3/s'),
        ensemble_quantile(ensemble, 0.25, 'flow_25%_m^3/s'),
        ensemble_quantile(ensemble, 0.00, 'flow_min_m^3/s'),
        high_res_df
    ], axis=1)
    return(stats_df)



def get_corrected_forecast(simulated_df, ensemble_df, observed_df):
    monthly_simulated = simulated_df[simulated_df.index.month == (ensemble_df.index[0]).month].dropna()
    monthly_observed = observed_df[observed_df.index.month == (ensemble_df.index[0]).month].dropna()
    min_simulated = np.min(monthly_simulated.iloc[:, 0].to_list())
    max_simulated = np.max(monthly_simulated.iloc[:, 0].to_list())
    min_factor_df = ensemble_df.copy()
    max_factor_df = ensemble_df.copy()
    forecast_ens_df = ensemble_df.copy()
    for column in ensemble_df.columns:
      tmp = ensemble_df[column].dropna().to_frame()
      min_factor = tmp.copy()
      max_factor = tmp.copy()
      min_factor.loc[min_factor[column] >= min_simulated, column] = 1
      min_index_value = min_factor[min_factor[column] != 1].index.tolist()
      for element in min_index_value:
        min_factor[column].loc[min_factor.index == element] = tmp[column].loc[tmp.index == element] / min_simulated
      max_factor.loc[max_factor[column] <= max_simulated, column] = 1
      max_index_value = max_factor[max_factor[column] != 1].index.tolist()
      for element in max_index_value:
        max_factor[column].loc[max_factor.index == element] = tmp[column].loc[tmp.index == element] / max_simulated
      tmp.loc[tmp[column] <= min_simulated, column] = min_simulated
      tmp.loc[tmp[column] >= max_simulated, column] = max_simulated
      forecast_ens_df.update(pd.DataFrame(tmp[column].values, index=tmp.index, columns=[column]))
      min_factor_df.update(pd.DataFrame(min_factor[column].values, index=min_factor.index, columns=[column]))
      max_factor_df.update(pd.DataFrame(max_factor[column].values, index=max_factor.index, columns=[column]))
    corrected_ensembles = geoglows.bias.correct_forecast(forecast_ens_df, simulated_df, observed_df)
    corrected_ensembles = corrected_ensembles.multiply(min_factor_df, axis=0)
    corrected_ensembles = corrected_ensembles.multiply(max_factor_df, axis=0)
    return(corrected_ensembles)



def get_corrected_forecast_records(records_df, simulated_df, observed_df):
    ''' Este es el comentario de la doc '''
    date_ini = records_df.index[0]
    month_ini = date_ini.month
    date_end = records_df.index[-1]
    month_end = date_end.month
    meses = np.arange(month_ini, month_end + 1, 1)
    fixed_records = pd.DataFrame()
    for mes in meses:
        values = records_df.loc[records_df.index.month == mes]
        monthly_simulated = simulated_df[simulated_df.index.month == mes].dropna()
        monthly_observed = observed_df[observed_df.index.month == mes].dropna()
        min_simulated = np.min(monthly_simulated.iloc[:, 0].to_list())
        max_simulated = np.max(monthly_simulated.iloc[:, 0].to_list())
        min_factor_records_df = values.copy()
        max_factor_records_df = values.copy()
        fixed_records_df = values.copy()
        column_records = values.columns[0]
        tmp = records_df[column_records].dropna().to_frame()
        min_factor = tmp.copy()
        max_factor = tmp.copy()
        min_factor.loc[min_factor[column_records] >= min_simulated, column_records] = 1
        min_index_value = min_factor[min_factor[column_records] != 1].index.tolist()
        for element in min_index_value:
            min_factor[column_records].loc[min_factor.index == element] = tmp[column_records].loc[tmp.index == element] / min_simulated
        max_factor.loc[max_factor[column_records] <= max_simulated, column_records] = 1
        max_index_value = max_factor[max_factor[column_records] != 1].index.tolist()
        for element in max_index_value:
            max_factor[column_records].loc[max_factor.index == element] = tmp[column_records].loc[tmp.index == element] / max_simulated
        tmp.loc[tmp[column_records] <= min_simulated, column_records] = min_simulated
        tmp.loc[tmp[column_records] >= max_simulated, column_records] = max_simulated
        fixed_records_df.update(pd.DataFrame(tmp[column_records].values, index=tmp.index, columns=[column_records]))
        min_factor_records_df.update(pd.DataFrame(min_factor[column_records].values, index=min_factor.index, columns=[column_records]))
        max_factor_records_df.update(pd.DataFrame(max_factor[column_records].values, index=max_factor.index, columns=[column_records]))
        corrected_values = geoglows.bias.correct_forecast(fixed_records_df, simulated_df, observed_df)
        corrected_values = corrected_values.multiply(min_factor_records_df, axis=0)
        corrected_values = corrected_values.multiply(max_factor_records_df, axis=0)
        fixed_records = fixed_records.append(corrected_values)
    fixed_records.sort_index(inplace=True)
    return(fixed_records)


def get_forecast_date(comid, date):
    url = 'https://geoglows.ecmwf.int/api/ForecastEnsembles/?reach_id={0}&date={1}&return_format=csv'.format(comid, date)
    status = False
    while not status:
      try:
        outdf = pd.read_csv(url, index_col=0)
        status = True
      except:
        print("Trying to retrieve data...")
    # Filter and correct data
    outdf[outdf < 0] = 0
    outdf.index = pd.to_datetime(outdf.index)
    outdf.index = outdf.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
    outdf.index = pd.to_datetime(outdf.index)
    return(outdf)


def get_forecast_record_date(comid, date):
    idate = dt.datetime.strptime(date, '%Y%m%d') - dt.timedelta(days=10)
    idate = idate.strftime('%Y%m%d')
    url = 'https://geoglows.ecmwf.int/api/ForecastRecords/?reach_id={0}&start_date={1}&end_date={2}&return_format=csv'.format(comid, idate, date)
    status = False
    while not status:
      try:
        outdf = pd.read_csv(url, index_col=0)
        status = True
      except:
        print("Trying to retrieve data...")
    # Filter and correct data
    outdf[outdf < 0] = 0
    outdf.index = pd.to_datetime(outdf.index)
    outdf.index = outdf.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
    outdf.index = pd.to_datetime(outdf.index)
    return(outdf)



def get_sonic_historical(comid, folder):
  forecast_nc_list = sorted(glob(os.path.join(folder, "*.nc")), reverse=True)
  nc_file = forecast_nc_list[0]
  qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_hist
  time_dataset = qout_datasets.time_hist
  historical_simulation_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Observed Streamflow'])
  historical_simulation_df.index = pd.to_datetime(historical_simulation_df.index)
  historical_simulation_df.index = historical_simulation_df.index.to_series().dt.strftime("%Y-%m-%d")
  historical_simulation_df.index = pd.to_datetime(historical_simulation_df.index)
  historical_simulation_df.index.name = 'datetime'
  return(historical_simulation_df)


def get_gfs_data(comid, initial_condition, folder, date=""):
  try:
    nc_file = "{0}/PISCO_HyD_ARNOVIC_v1.0_{1}.nc".format(folder, date)
    qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_gfs
    time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_frst
  except:
    forecast_nc_list = sorted(glob(os.path.join(folder, "*.nc")), reverse=True)
    nc_file = forecast_nc_list[0]
    qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_gfs
    time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_frst
  forecast_gfs_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
  forecast_gfs_df.index.name = 'datetime'
  forecast_gfs_df = forecast_gfs_df.append(initial_condition)
  forecast_gfs_df.sort_index(inplace=True)
  forecast_gfs_df.index = pd.to_datetime(forecast_gfs_df.index)
  forecast_gfs_df.index = forecast_gfs_df.index.to_series().dt.strftime("%Y-%m-%d")
  forecast_gfs_df.index = pd.to_datetime(forecast_gfs_df.index)
  return(forecast_gfs_df)


def get_eta_eqm_data(comid, initial_condition, folder, date=""):
  try:
    nc_file = "{0}/PISCO_HyD_ARNOVIC_v1.0_{1}.nc".format(folder, date)
    qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_eta_eqm
    time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_frst
  except:
    forecast_nc_list = sorted(glob(os.path.join(folder, "*.nc")), reverse=True)
    nc_file = forecast_nc_list[0]
    qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_eta_eqm
    time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_frst
  forecast_eta_eqm_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
  forecast_eta_eqm_df.index.name = 'datetime'
  forecast_eta_eqm_df = forecast_eta_eqm_df.append(initial_condition)
  forecast_eta_eqm_df.sort_index(inplace=True)
  forecast_eta_eqm_df.index = pd.to_datetime(forecast_eta_eqm_df.index)
  forecast_eta_eqm_df.index = forecast_eta_eqm_df.index.to_series().dt.strftime("%Y-%m-%d")
  forecast_eta_eqm_df.index = pd.to_datetime(forecast_eta_eqm_df.index)
  return(forecast_eta_eqm_df)


def get_eta_scal_data(comid, initial_condition, folder, date=""):
  try:
    nc_file = "{0}/PISCO_HyD_ARNOVIC_v1.0_{1}.nc".format(folder, date)
    qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_eta_scal
    time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_frst
  except:
    forecast_nc_list = sorted(glob(os.path.join(folder, "*.nc")), reverse=True)
    nc_file = forecast_nc_list[0]
    qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_eta_scal
    time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_frst
  forecast_eta_scal_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
  forecast_eta_scal_df.index.name = 'datetime'
  forecast_eta_scal_df = forecast_eta_scal_df.append(initial_condition)
  forecast_eta_scal_df.sort_index(inplace=True)
  forecast_eta_scal_df.index = pd.to_datetime(forecast_eta_scal_df.index)
  forecast_eta_scal_df.index = forecast_eta_scal_df.index.to_series().dt.strftime("%Y-%m-%d")
  forecast_eta_scal_df.index = pd.to_datetime(forecast_eta_scal_df.index)
  return(forecast_eta_scal_df)


def get_wrf_data(comid, initial_condition, folder, date=""):
  try:
    nc_file = "{0}/PISCO_HyD_ARNOVIC_v1.0_{1}.nc".format(folder, date)
    qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_wrf
    time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_frst
  except:
    forecast_nc_list = sorted(glob(os.path.join(folder, "*.nc")), reverse=True)
    nc_file = forecast_nc_list[0]
    qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_wrf
    time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_frst
  forecast_wrf_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
  forecast_wrf_df.index.name = 'datetime'
  forecast_wrf_df = forecast_wrf_df.append(initial_condition)
  forecast_wrf_df.sort_index(inplace=True)
  forecast_wrf_df.index = pd.to_datetime(forecast_wrf_df.index)
  forecast_wrf_df.index = forecast_wrf_df.index.to_series().dt.strftime("%Y-%m-%d")
  forecast_wrf_df.index = pd.to_datetime(forecast_wrf_df.index)
  return(forecast_wrf_df)