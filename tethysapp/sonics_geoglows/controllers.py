####################################################################################################
##                                   LIBRARIES AND DEPENDENCIES                                   ##
####################################################################################################

# Tethys platform
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from tethys_sdk.routing import controller
from tethys_sdk.gizmos import PlotlyView

# Postgresql
import io
import os
import datetime as dt
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from pandas_geojson import to_geojson
from glob import glob

# App settings
from .app import SonicsGeoglows as app

# App models
from .models.data import *
from .models.plot import *



####################################################################################################
##                                       STATUS VARIABLES                                         ##
####################################################################################################

# Import enviromental variables 
DB_USER = app.get_custom_setting('DB_USER')
DB_PASS = app.get_custom_setting('DB_PASS')
DB_NAME = app.get_custom_setting('DB_NAME')
FOLDER =  app.get_custom_setting('FOLDER')

# Generate the conection token
tokencon = "postgresql+psycopg2://{0}:{1}@localhost:5432/{2}".format(DB_USER, DB_PASS, DB_NAME)



####################################################################################################
##                                   CONTROLLERS AND REST APIs                                    ##
####################################################################################################

@controller(name='home',url='sonics-geoglows/')
def home(request):
    forecast_nc_list = sorted(glob(os.path.join(FOLDER, "*.nc")))
    dates_array = []
    for file in forecast_nc_list:
        dates_array.append(file[len(FOLDER) + 1 + 23:-3])
    dates = []
    for date in dates_array:
        date_f = dt.datetime(int(date[0:4]), int(date[4:6]), int(date[6:8])).strftime('%Y-%m-%d')
        dates.append([date_f, date])
    context = {
        "server": app.get_custom_setting('SERVER'),
        "app_name": app.package,
        "start_date": dates[0][0],
        "end_date": dates[-1][0]
    }
    return render(request, 'sonics_geoglows/home.html', context) 


# Return alerts (in geojson format)
@controller(name='get_alerts',url='sonics-geoglows/get-alerts')
def get_alerts(request):
    # Establish connection to database
    db = create_engine(tokencon)
    conn = db.connect()
    # Query to database
    stations = pd.read_sql("select * from sonics_geoglows where alert != 'R0'", conn);
    conn.close()
    stations = to_geojson(
        df = stations,
        lat = "latitude",
        lon = "longitude",
        properties = ["comid", "latitude", "longitude", "loc1", "loc2", "alert"]
    )
    return JsonResponse(stations)


# Return rivers (in geojson format)
@controller(name='get_rivers',url='sonics-geoglows/get-rivers')
def get_rivers(request):
    # Establish connection to database
    db = create_engine(tokencon)
    conn = db.connect()
    # Query to database
    stations = pd.read_sql("select comid, latitude, longitude from sonics_geoglows", conn);
    conn.close()
    stations = to_geojson(
        df = stations,
        lat = "latitude",
        lon = "longitude",
        properties = ["comid", "latitude", "longitude"]
    )
    return JsonResponse(stations)



# Return streamflow station (in geojson format) 
@controller(name='get_data',url='sonics-geoglows/get-data')
def get_data(request):
    # Retrieving GET arguments
    station_comid = request.GET['comid']
    station_code = str(station_comid)
    forecast_date = request.GET['fecha']
    plot_width = float(request.GET['width']) - 12
    plot_width_2 = 0.5*plot_width

    # Establish connection to database
    db= create_engine(tokencon)
    conn = db.connect()

    # Data series
    observed_data = get_sonic_historical(station_comid, FOLDER)
    simulated_data = get_format_data("select * from r_{0} where datetime < '2022-06-01 00:00:00';".format(station_comid), conn)
    corrected_data = get_bias_corrected_data(simulated_data, observed_data)

    # Raw forecast
    #ensemble_forecast = get_format_data("select * from f_{0};".format(station_comid), conn)
    ensemble_forecast = get_forecast_date(station_comid, forecast_date)
    forecast_records = get_format_data("select * from fr_{0};".format(station_comid), conn)
    return_periods = get_return_periods(station_comid, simulated_data)

    # Corrected forecast
    corrected_ensemble_forecast = get_corrected_forecast(simulated_data, ensemble_forecast, observed_data)
    corrected_forecast_records = get_corrected_forecast_records(forecast_records, simulated_data, observed_data)
    corrected_return_periods = get_return_periods(station_comid, corrected_data)

    # Stats for raw and corrected forecast
    ensemble_stats = get_ensemble_stats(ensemble_forecast)
    corrected_ensemble_stats = get_ensemble_stats(corrected_ensemble_forecast)

    # Merge data (For plots)
    global merged_sim
    merged_sim = hd.merge_data(sim_df = simulated_data, obs_df = observed_data)
    global merged_cor
    merged_cor = hd.merge_data(sim_df = corrected_data, obs_df = observed_data)

    # Close conection
    conn.close()

    # SONICS forecast
    initial_condition = observed_data.loc[observed_data.index == pd.to_datetime(observed_data.index[-1])]
    initial_condition.index = pd.to_datetime(initial_condition.index)
    initial_condition.index = initial_condition.index.to_series().dt.strftime("%Y-%m-%d")
    initial_condition.index = pd.to_datetime(initial_condition.index)
    initial_condition.rename(columns = {'Observed Streamflow':'Streamflow (m3/s)'}, inplace = True)
    gfs_data = get_gfs_data(station_comid, initial_condition, FOLDER)
    eta_eqm_data = get_eta_eqm_data(station_comid, initial_condition, FOLDER)
    eta_scal_data = get_eta_scal_data(station_comid, initial_condition, FOLDER)
    wrf_data = get_wrf_data(station_comid, initial_condition, FOLDER)

    # Historical data plot
    corrected_data_plot = plot_historical(
                                observed_df = observed_data, 
                                simulated_df = simulated_data, 
                                corrected_df = corrected_data, 
                                station_code = station_code)
    
    # Daily averages plot
    daily_average_plot = get_daily_average_plot(
                                merged_cor = merged_cor,
                                merged_sim = merged_sim,
                                code = station_code)   
    # Monthly averages plot
    monthly_average_plot = get_monthly_average_plot(
                                merged_cor = merged_cor,
                                merged_sim = merged_sim,
                                code = station_code) 
    # Scatter plot
    data_scatter_plot = get_scatter_plot(
                                merged_cor = merged_cor,
                                merged_sim = merged_sim,
                                code = station_code,
                                log = False) 
    # Scatter plot (Log scale)
    log_data_scatter_plot = get_scatter_plot(
                                merged_cor = merged_cor,
                                merged_sim = merged_sim,
                                code = station_code,
                                log = True) 
    
    # Metrics table
    metrics_table = get_metrics_table(
                                merged_cor = merged_cor,
                                merged_sim = merged_sim,
                                my_metrics = ["ME", "RMSE", "NRMSE (Mean)", "NSE", "KGE (2009)", "KGE (2012)", "R (Pearson)", "R (Spearman)", "r2"]) 
    
    # Ensemble forecast plot    
    ensemble_forecast_plot = get_forecast_plot(
                                comid = station_comid,  
                                stats = ensemble_stats, 
                                rperiods = return_periods, 
                                records = forecast_records,
                                historical_sonics = observed_data,
                                gfs = gfs_data,
                                eta_eqm = eta_eqm_data,
                                eta_scal = eta_scal_data,
                                wrf = wrf_data)

    # Ensemble forecast plot    
    corrected_ensemble_forecast_plot = get_forecast_plot(
                                            comid = station_comid,  
                                            stats = corrected_ensemble_stats, 
                                            rperiods = corrected_return_periods, 
                                            records = corrected_forecast_records,
                                            historical_sonics = observed_data,
                                            gfs = gfs_data,
                                            eta_eqm = eta_eqm_data,
                                            eta_scal = eta_scal_data,
                                            wrf = wrf_data)
    
    #returning
    context = {
        "corrected_data_plot": PlotlyView(corrected_data_plot.update_layout(width = plot_width)),
        "daily_average_plot": PlotlyView(daily_average_plot.update_layout(width = plot_width)),
        "monthly_average_plot": PlotlyView(monthly_average_plot.update_layout(width = plot_width)),
        "data_scatter_plot": PlotlyView(data_scatter_plot.update_layout(width = plot_width_2)),
        "log_data_scatter_plot": PlotlyView(log_data_scatter_plot.update_layout(width = plot_width_2)),
        "ensemble_forecast_plot": PlotlyView(ensemble_forecast_plot.update_layout(width = plot_width)),
        "corrected_ensemble_forecast_plot": PlotlyView(corrected_ensemble_forecast_plot.update_layout(width = plot_width)),
        "metrics_table": metrics_table,
    }
    return render(request, 'sonics_geoglows/panel.html', context)





@controller(name='get_raw_forecast_date',url='sonics-geoglows/get-raw-forecast-date')
def get_raw_forecast_date(request):
    ## Variables
    station_comid = request.GET['comid']
    forecast_date = request.GET['fecha']
    plot_width = float(request.GET['width']) - 12

    # Establish connection to database
    db= create_engine(tokencon)
    conn = db.connect()

    # Data series
    observed_data = get_sonic_historical(station_comid, FOLDER)
    simulated_data = get_format_data("select * from r_{0} where datetime < '2022-06-01 00:00:00';".format(station_comid), conn)
    corrected_data = get_bias_corrected_data(simulated_data, observed_data)
    
    # Raw forecast
    ensemble_forecast = get_forecast_date(station_comid, forecast_date)
    forecast_records = get_format_data("select * from fr_{0};".format(station_comid), conn)
    return_periods = get_return_periods(station_comid, simulated_data)

    # Corrected forecast
    corrected_ensemble_forecast = get_corrected_forecast(simulated_data, ensemble_forecast, observed_data)
    corrected_forecast_records = get_corrected_forecast_records(forecast_records, simulated_data, observed_data)
    corrected_return_periods = get_return_periods(station_comid, corrected_data)
    
    # Forecast stats
    ensemble_stats = get_ensemble_stats(ensemble_forecast)
    corrected_ensemble_stats = get_ensemble_stats(corrected_ensemble_forecast)

    # Close conection
    conn.close()

    # SONICS forecast
    initial_condition = observed_data.loc[observed_data.index == pd.to_datetime(observed_data.index[-1])]
    initial_condition.index = pd.to_datetime(initial_condition.index)
    initial_condition.index = initial_condition.index.to_series().dt.strftime("%Y-%m-%d")
    initial_condition.index = pd.to_datetime(initial_condition.index)
    initial_condition.rename(columns = {'Observed Streamflow':'Streamflow (m3/s)'}, inplace = True)
    gfs_data = get_gfs_data(station_comid, initial_condition, FOLDER, forecast_date)
    eta_eqm_data = get_eta_eqm_data(station_comid, initial_condition, FOLDER, forecast_date)
    eta_scal_data = get_eta_scal_data(station_comid, initial_condition, FOLDER, forecast_date)
    wrf_data = get_wrf_data(station_comid, initial_condition, FOLDER, forecast_date)
    
    # Plotting raw forecast
    ensemble_forecast_plot = get_forecast_plot(
                                comid = station_comid,  
                                stats = ensemble_stats, 
                                rperiods = return_periods, 
                                records = forecast_records,
                                historical_sonics = observed_data,
                                gfs = gfs_data,
                                eta_eqm = eta_eqm_data,
                                eta_scal = eta_scal_data,
                                wrf = wrf_data).update_layout(width = plot_width).to_html()


    # Plotting corrected forecast
    corr_ensemble_forecast_plot = get_forecast_plot(
                                            comid = station_comid,  
                                            stats = corrected_ensemble_stats, 
                                            rperiods = corrected_return_periods, 
                                            records = corrected_forecast_records,
                                            historical_sonics = observed_data,
                                            gfs = gfs_data,
                                            eta_eqm = eta_eqm_data,
                                            eta_scal = eta_scal_data,
                                            wrf = wrf_data).update_layout(width = plot_width).to_html()
    
    return JsonResponse({
       'ensemble_forecast_plot': ensemble_forecast_plot,
       'corr_ensemble_forecast_plot': corr_ensemble_forecast_plot,
    })
    




# Retrieve xlsx data
@controller(name='get_simulated_data_sonics_xlsx',url='sonics-geoglows/get-observed-data-xlsx')
def get_simulated_data_sonics_xlsx(request):
    # Retrieving GET arguments
    station_comid = request.GET['comid'] #9027406
    # Data series
    observed_data = get_sonic_historical(station_comid, FOLDER)
    observed_data = observed_data.rename(columns={"Observed Streamflow": "SONICS historical simulation (m3/s)"})
    # Crear el archivo Excel
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    observed_data.to_excel(writer, sheet_name='serie_historica_sonics', index=True)  # Aquí se incluye el índice 
    writer.save()
    output.seek(0)
    # Configurar la respuesta HTTP para descargar el archivo
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=serie_historica_sonics.xlsx'
    response.write(output.getvalue())
    return response


# Retrieve xlsx data
@controller(name='get_simulated_data_geoglows_xlsx',url='sonics-geoglows/get-simulated-data-xlsx')
def get_simulated_data_geoglows_xlsx(request):
    # Retrieving GET arguments
    station_comid = request.GET['comid'] #9027406
    # Establish connection to database
    db= create_engine(tokencon)
    conn = db.connect()
    # Data series
    simulated_data = get_format_data("select * from r_{0} where datetime < '2022-06-01 00:00:00';".format(station_comid), conn)
    simulated_data = simulated_data.rename(columns={"streamflow_m^3/s": "GEOGloWS historical simulation (m3/s)"})
    # Crear el archivo Excel
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    simulated_data.to_excel(writer, sheet_name='serie_historica_geoglows', index=True)  # Aquí se incluye el índice
    writer.save()
    output.seek(0)
    # Configurar la respuesta HTTP para descargar el archivo
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=serie_historica_geoglows.xlsx'
    response.write(output.getvalue())
    return response


# Retrieve xlsx data
@controller(name='get_corrected_data_xlsx',url='sonics-geoglows/get-corrected-data-xlsx')
def get_corrected_data_xlsx(request):
    # Retrieving GET arguments
    station_comid = request.GET['comid'] #9027406
    # Establish connection to database
    db= create_engine(tokencon)
    conn = db.connect()
    # Data series
    observed_data = get_sonic_historical(station_comid, FOLDER)
    simulated_data = get_format_data("select * from r_{0} where datetime < '2022-06-01 00:00:00';".format(station_comid), conn)
    corrected_data = get_bias_corrected_data(simulated_data, observed_data)
    corrected_data = corrected_data.rename(columns={"Corrected Simulated Streamflow" : "Corrected Simulated Streamflow (m3/s)"})
    # Crear el archivo Excel
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    corrected_data.to_excel(writer, sheet_name='serie_historica_corregida', index=True)  # Aquí se incluye el índice
    writer.save()
    output.seek(0)
    # Configurar la respuesta HTTP para descargar el archivo
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=serie_historica_corregida.xlsx'
    response.write(output.getvalue())
    return response



# Retrieve xlsx data
@controller(name='get_sonics_forecast_xlsx',url='sonics-geoglows/get-sonics-xlsx')
def get_sonics_forecast_xlsx(request):
    # Retrieving GET arguments
    station_comid = request.GET['comid'] #9027406
    forecast_date = request.GET['fecha']
    # Data series
    observed_data = get_sonic_historical(station_comid, FOLDER)
    initial_condition = observed_data.loc[observed_data.index == pd.to_datetime(observed_data.index[-1])]
    initial_condition.index = pd.to_datetime(initial_condition.index)
    initial_condition.index = initial_condition.index.to_series().dt.strftime("%Y-%m-%d")
    initial_condition.index = pd.to_datetime(initial_condition.index)
    initial_condition.rename(columns = {'Observed Streamflow':'Streamflow (m3/s)'}, inplace = True)
    gfs_data = get_gfs_data(station_comid, initial_condition, FOLDER, forecast_date)
    gfs_data.rename(columns = {'Streamflow (m3/s)':'GFS (m3/S)'}, inplace = True)
    eta_eqm_data = get_eta_eqm_data(station_comid, initial_condition, FOLDER, forecast_date)
    eta_eqm_data.rename(columns = {'Streamflow (m3/s)':'ETA EQM (m3/S)'}, inplace = True)
    eta_scal_data = get_eta_scal_data(station_comid, initial_condition, FOLDER, forecast_date)
    eta_scal_data.rename(columns = {'Streamflow (m3/s)':'ETA SCAL (m3/S)'}, inplace = True)
    wrf_data = get_wrf_data(station_comid, initial_condition, FOLDER, forecast_date)
    wrf_data.rename(columns = {'Streamflow (m3/s)':'WRF (m3/S)'}, inplace = True)
    # Combined data series
    sonics_forecast = gfs_data.merge(eta_eqm_data, on='datetime', how='inner')
    sonics_forecast = sonics_forecast.merge(eta_scal_data, on='datetime', how='inner')
    sonics_forecast = sonics_forecast.merge(wrf_data, on='datetime', how='inner')
    # Crear el archivo Excel
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    sonics_forecast.to_excel(writer, sheet_name='pronostico_sonics', index=True)  # Aquí se incluye el índice
    writer.save()
    output.seek(0)
    # Configurar la respuesta HTTP para descargar el archivo
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=pronostico_sonics.xlsx'
    response.write(output.getvalue())
    return response



# Retrieve xlsx data
@controller(name='get_forecast_xlsx',url='sonics-geoglows/get-forecast-xlsx')
def get_forecast_xlsx(request):
    # Retrieving GET arguments
    station_comid = request.GET['comid']
    forecast_date = request.GET['fecha']
    # Establish connection to database
    db= create_engine(tokencon)
    conn = db.connect()
    # Raw forecast
    ensemble_forecast = get_forecast_date(station_comid, forecast_date)
    ensemble_stats = get_ensemble_stats(ensemble_forecast)
    # Crear el archivo Excel
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    ensemble_stats.to_excel(writer, sheet_name='ensemble_forecast', index=True)  # Aquí se incluye el índice
    writer.save()
    output.seek(0)
    # Configurar la respuesta HTTP para descargar el archivo
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=ensemble_forecast.xlsx'
    response.write(output.getvalue())
    return response




@controller(name='get_corrected_forecast_xlsx',url='sonics-geoglows/get-corrected-forecast-xlsx')
def get_corrected_forecast_xlsx(request):
    # Retrieving GET arguments
    station_comid = request.GET['comid']
    forecast_date = request.GET['fecha']
    # Establish connection to database
    db= create_engine(tokencon)
    conn = db.connect()
    # Data series
    observed_data = get_sonic_historical(station_comid, FOLDER)
    simulated_data = get_format_data("select * from r_{0} where datetime < '2022-06-01 00:00:00';".format(station_comid), conn)
    # Raw forecast
    ensemble_forecast = get_forecast_date(station_comid, forecast_date)
    # Corrected forecast
    corrected_ensemble_forecast = get_corrected_forecast(simulated_data, ensemble_forecast, observed_data)
    # Forecast stats
    corrected_ensemble_stats = get_ensemble_stats(corrected_ensemble_forecast)
    # Crear el archivo Excel
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    corrected_ensemble_stats.to_excel(writer, sheet_name='corrected_ensemble_forecast', index=True)  # Aquí se incluye el índice
    writer.save()
    output.seek(0)
    # Configurar la respuesta HTTP para descargar el archivo
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=corrected_ensemble_forecast.xlsx'
    response.write(output.getvalue())
    return response
