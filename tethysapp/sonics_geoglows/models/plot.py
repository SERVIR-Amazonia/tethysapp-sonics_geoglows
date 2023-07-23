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
from plotly.offline import plot as offline_plot


####################################################################################################
##                                      PLOTTING FUNCTIONS                                        ##
####################################################################################################

# Historical data
def plot_historical(observed_df, simulated_df, corrected_df, station_code):
    observed_plot = go.Scatter(x=observed_df.index, y=observed_df.iloc[:, 0].values, name='SONICS', line=dict(color="#636EFA"))
    simulated_plot = go.Scatter(x=simulated_df.index, y=simulated_df.iloc[:, 0].values, name='GEOGloWS', line=dict(color="#EF553B"))
    corrected_plot = go.Scatter(x=corrected_df.index, y=corrected_df.iloc[:, 0].values, name='GEOGloWS corregido', line=dict(color="#00CC96"))
    layout = go.Layout(
            title='Simulación histórica:<br>COMID: {0}'.format(station_code),
            xaxis=dict(title='Serie temporal', ), yaxis=dict(title='Caudal (m<sup>3</sup>/s)', autorange=True),
            showlegend=True)
    return(go.Figure(data=[observed_plot, simulated_plot, corrected_plot], layout=layout))


# Plotting daily averages values
def get_daily_average_plot(merged_sim, merged_cor, code):
    # Generate the average values
    daily_avg_sim = hd.daily_average(merged_sim)
    daily_avg_cor = hd.daily_average(merged_cor)
    # Generate the plots on Ploty
    daily_avg_obs_Q = go.Scatter(x = daily_avg_sim.index, y = daily_avg_sim.iloc[:, 1].values, name = 'SONICS', )
    daily_avg_sim_Q = go.Scatter(x = daily_avg_sim.index, y = daily_avg_sim.iloc[:, 0].values, name = 'GEOGloWS', )
    daily_avg_corr_sim_Q = go.Scatter(x = daily_avg_cor.index, y = daily_avg_cor.iloc[:, 0].values, name = 'GEOGloWS corregido', )
    # PLot Layout
    layout = go.Layout(
        title='Caudal promedio multi-diario <br>COMID: {0}'.format(str(code).upper()),
        xaxis=dict(title='Día', ), 
        yaxis=dict(title='Caudal (m<sup>3</sup>/s)', autorange=True),
        showlegend=True)
    # Generate the output
    chart_obj = go.Figure(data=[daily_avg_obs_Q, daily_avg_sim_Q, daily_avg_corr_sim_Q], layout=layout)
    return(chart_obj)


# Plotting monthly averages values
def get_monthly_average_plot(merged_sim, merged_cor, code):
    # Generate the average values
    daily_avg_sim = hd.monthly_average(merged_sim)
    daily_avg_cor = hd.monthly_average(merged_cor)
    # Generate the plots on Ploty
    daily_avg_obs_Q = go.Scatter(x = daily_avg_sim.index, y = daily_avg_sim.iloc[:, 1].values, name = 'SONICS', )
    daily_avg_sim_Q = go.Scatter(x = daily_avg_sim.index, y = daily_avg_sim.iloc[:, 0].values, name = 'GEOGloWS', )
    daily_avg_corr_sim_Q = go.Scatter(x = daily_avg_cor.index, y = daily_avg_cor.iloc[:, 0].values, name = 'GEOGloWS corregido', )
    # PLot Layout
    layout = go.Layout(
        title='Caudal promedio multi-mensual <br>COMID: {0}'.format(str(code).upper()),
        xaxis=dict(title='Mes', ), 
        yaxis=dict(title='Caudal (m<sup>3</sup>/s)', autorange=True),
        showlegend=True)
    # Generate the output
    chart_obj = go.Figure(data=[daily_avg_obs_Q, daily_avg_sim_Q, daily_avg_corr_sim_Q], layout=layout)
    return(chart_obj)





# Scatter plot (Simulated/Corrected vs Observed)
def get_scatter_plot(merged_sim, merged_cor, code, log):
    # Generate Scatter (sim vs obs)
    scatter_data = go.Scatter(
        x = merged_sim.iloc[:, 0].values,
        y = merged_sim.iloc[:, 1].values,
        mode='markers',
        name='Original',
        marker=dict(color='#ef553b'))
    # Generate Scatter (cor vs obs)
    scatter_data2 = go.Scatter(
        x = merged_cor.iloc[:, 0].values,
        y = merged_cor.iloc[:, 1].values,
        mode='markers',
        name='Corregido',
        marker=dict(color='#00cc96'))
    # Get the max and min values
    min_value = min(min(merged_sim.iloc[:, 1].values), min(merged_sim.iloc[:, 0].values))
    max_value = max(max(merged_sim.iloc[:, 1].values), max(merged_sim.iloc[:, 0].values))
    # Construct the line 1:1
    line_45 = go.Scatter(
        x = [min_value, max_value],
        y = [min_value, max_value],
        mode = 'lines',
        name = 'Linea 1:1',
        line = dict(color='black'))
    # Plot Layout
    if log == True:
        layout = go.Layout(title = "Diagrama de dispersión logarítmica (obs vs sim)<br>COMID:{0}".format(str(code).upper()),
                       xaxis = dict(title = 'Caudal simulado (m<sup>3</sup>/s)', type = 'log', ), 
                       yaxis = dict(title = 'Caudal observado (m<sup>3</sup>/s)', type = 'log', autorange = True), 
                       showlegend=True)
    else:
        layout = go.Layout(title = "Diagrama de dispersión (obs vs sim)<br>COMID: {0}".format(str(code).upper()),
                       xaxis = dict(title = 'Caudal simulado (m<sup>3</sup>/s)',  ), 
                       yaxis = dict(title = 'Caudal observado (m<sup>3</sup>/s)', autorange = True), 
                       showlegend=True)
    # Plotting data
    chart_obj = go.Figure(data=[scatter_data, scatter_data2, line_45], layout=layout)
    return(chart_obj)




# Metrics table
def get_metrics_table(merged_sim, merged_cor, my_metrics):
    # Metrics for simulated data
    table_sim = hs.make_table(merged_sim, my_metrics)
    table_sim = table_sim.rename(index={'Full Time Series': 'Simulación GEOGloWS'})
    table_sim = table_sim.transpose()
    # Metrics for corrected simulation data
    table_cor = hs.make_table(merged_cor, my_metrics)
    table_cor = table_cor.rename(index={'Full Time Series': 'Simulación GEOGloWS corregida'})
    table_cor = table_cor.transpose()
    # Merging data
    table_final = pd.merge(table_sim, table_cor, right_index=True, left_index=True)
    table_final = table_final.round(decimals=2)
    table_final = table_final.to_html(classes="table table-hover table-striped", table_id="corrected_1")
    table_final = table_final.replace('border="1"', 'border="0"').replace('<tr style="text-align: right;">','<tr style="text-align: left;">')
    return(table_final)


######################
######################
def _build_title(base, title_headers):
    if not title_headers:
        return base
    if 'bias_corrected' in title_headers.keys():
        base = 'Bias Corrected ' + base
    for head in title_headers:
        if head == 'bias_corrected':
            continue
        base += f'<br>{head}: {title_headers[head]}'
    return base

def _plot_colors():
    return {
        '2 Year': 'rgba(254, 240, 1, .4)',
        '5 Year': 'rgba(253, 154, 1, .4)',
        '10 Year': 'rgba(255, 56, 5, .4)',
        '20 Year': 'rgba(128, 0, 246, .4)',
        '25 Year': 'rgba(255, 0, 0, .4)',
        '50 Year': 'rgba(128, 0, 106, .4)',
        '100 Year': 'rgba(128, 0, 246, .4)',
    }

def _rperiod_scatters(startdate: str, enddate: str, rperiods: pd.DataFrame, y_max: float, max_visible: float = 0):
    colors = _plot_colors()
    x_vals = (startdate, enddate, enddate, startdate)
    r2 = int(rperiods['return_period_2'].values[0])
    if max_visible > r2:
        visible = True
    else:
        visible = 'legendonly'

    def template(name, y, color, fill='toself'):
        return go.Scatter(
            name=name,
            x=x_vals,
            y=y,
            legendgroup='returnperiods',
            fill=fill,
            visible=visible,
            line=dict(color=color, width=0))

    if list(rperiods.columns) == ['max_flow', 'return_period_20', 'return_period_10', 'return_period_2']:
        r10 = int(rperiods['return_period_10'].values[0])
        r20 = int(rperiods['return_period_20'].values[0])
        rmax = int(max(2 * r20 - r10, y_max))
        return [
            template(f'2 Year: {r2}', (r2, r2, r10, r10), colors['2 Year']),
            template(f'10 Year: {r10}', (r10, r10, r20, r20), colors['10 Year']),
            template(f'20 Year: {r20}', (r20, r20, rmax, rmax), colors['20 Year']),
        ]

    else:
        r5 = int(rperiods['return_period_5'].values[0])
        r10 = int(rperiods['return_period_10'].values[0])
        r25 = int(rperiods['return_period_25'].values[0])
        r50 = int(rperiods['return_period_50'].values[0])
        r100 = int(rperiods['return_period_100'].values[0])
        rmax = int(max(2 * r100 - r25, y_max))
        return [
            template('Return Periods', (rmax, rmax, rmax, rmax), 'rgba(0,0,0,0)', fill='none'),
            template(f'2 Year: {r2}', (r2, r2, r5, r5), colors['2 Year']),
            template(f'5 Year: {r5}', (r5, r5, r10, r10), colors['5 Year']),
            template(f'10 Year: {r10}', (r10, r10, r25, r25), colors['10 Year']),
            template(f'25 Year: {r25}', (r25, r25, r50, r50), colors['25 Year']),
            template(f'50 Year: {r50}', (r50, r50, r100, r100), colors['50 Year']),
            template(f'100 Year: {r100}', (r100, r100, rmax, rmax), colors['100 Year']),
        ]

def forecast_stats_es(stats: pd.DataFrame, rperiods: pd.DataFrame = None, titles: dict = False,
                   outformat: str = 'plotly') -> go.Figure:

    if outformat not in ['json', 'plotly_scatters', 'plotly', 'plotly_html']:
        raise ValueError('invalid outformat specified. pick json, plotly, plotly_scatters, or plotly_html')

    # Start processing the inputs
    dates = stats.index.tolist()
    startdate = dates[0]
    enddate = dates[-1]

    plot_data = {
        'x_stats': stats['flow_avg_m^3/s'].dropna(axis=0).index.tolist(),
        'x_hires': stats['high_res_m^3/s'].dropna(axis=0).index.tolist(),
        'y_max': max(stats['flow_max_m^3/s']),
        'flow_max': list(stats['flow_max_m^3/s'].dropna(axis=0)),
        'flow_75%': list(stats['flow_75%_m^3/s'].dropna(axis=0)),
        'flow_avg': list(stats['flow_avg_m^3/s'].dropna(axis=0)),
        'flow_25%': list(stats['flow_25%_m^3/s'].dropna(axis=0)),
        'flow_min': list(stats['flow_min_m^3/s'].dropna(axis=0)),
        'high_res': list(stats['high_res_m^3/s'].dropna(axis=0)),
    }
    if rperiods is not None:
        plot_data.update(rperiods.to_dict(orient='index').items())
        max_visible = max(max(plot_data['flow_75%']), max(plot_data['flow_avg']), max(plot_data['high_res']))
        rperiod_scatters = _rperiod_scatters(startdate, enddate, rperiods, plot_data['y_max'], max_visible)
    else:
        rperiod_scatters = []
    if outformat == 'json':
        return plot_data

    scatter_plots = [
        # Plot together so you can use fill='toself' for the shaded box, also separately so the labels appear
        go.Scatter(name='Máximos y mínimos',
                   x=plot_data['x_stats'] + plot_data['x_stats'][::-1],
                   y=plot_data['flow_max'] + plot_data['flow_min'][::-1],
                   legendgroup='boundaries',
                   fill='toself',
                   visible='legendonly',
                   line=dict(color='darkblue', dash='dash'),
                   fillcolor='lightblue'),
        go.Scatter(name='Percentiles 25-75',
                   x=plot_data['x_stats'] + plot_data['x_stats'][::-1],
                   y=plot_data['flow_75%'] + plot_data['flow_25%'][::-1],
                   legendgroup='percentile_flow',
                   fill='toself',
                   line=dict(color='lightgreen'), ),
        go.Scatter(name='75%',
                   x=plot_data['x_stats'],
                   y=plot_data['flow_75%'],
                   showlegend=False,
                   legendgroup='percentile_flow',
                   line=dict(color='green'), ),
        go.Scatter(name='25%',
                   x=plot_data['x_stats'],
                   y=plot_data['flow_25%'],
                   showlegend=False,
                   legendgroup='percentile_flow',
                   line=dict(color='green'), ),
        go.Scatter(name='Pronóstico de alta resolución',
                   x=plot_data['x_hires'],
                   y=plot_data['high_res'],
                   line={'color': 'black'}, ),
        go.Scatter(name='Promedio del ensamble',
                   x=plot_data['x_stats'],
                   y=plot_data['flow_avg'],
                   line=dict(color='blue'), ),
    ]
    scatter_plots += rperiod_scatters

    if outformat == 'plotly_scatters':
        return scatter_plots

    layout = go.Layout(
        title=_build_title('Pronóstico de caudales', titles),
        yaxis={'title': 'Caudal (m<sup>3</sup>/s)', 'range': [0, 'auto']},
        xaxis={'title': 'Fecha (UTC +0:00)', 'range': [startdate, enddate], 'hoverformat': '%b %d %Y',
               'tickformat': '%b %d %Y'},
    )
    figure = go.Figure(scatter_plots, layout=layout)
    if outformat == 'plotly':
        return figure
    if outformat == 'plotly_html':
        return offline_plot(
            figure,
            config={'autosizable': True, 'responsive': True},
            output_type='div',
            include_plotlyjs=False
        )
    return




# Forecast plot
def get_forecast_plot(comid, stats, rperiods, records, historical_sonics, gfs, eta_eqm, eta_scal, wrf):
    stats_df = stats
    forecast_gfs_df = gfs
    forecast_eta_eqm_df = eta_eqm
    forecast_eta_scal_df = eta_scal
    forecast_wrf_df = wrf
    rperiods_geoglows = rperiods
    #
    titles = {'COMID': comid}
    hydroviewer_figure = forecast_stats_es(stats=stats_df, titles=titles)
    #
    x_vals = (stats_df.index[0], stats_df.index[len(stats_df.index) - 1], stats_df.index[len(stats_df.index) - 1], stats_df.index[0])
    max_visible = max(stats_df.max())
    #
    geoglows_records = records.copy()
    geoglows_records = geoglows_records.loc[geoglows_records.index >= pd.to_datetime(stats_df.index[0] - dt.timedelta(days=8))]
    geoglows_records = geoglows_records.loc[geoglows_records.index <= pd.to_datetime(stats_df.index[0] + dt.timedelta(days=2))]
    #
    if len(geoglows_records.index) > 0:
        hydroviewer_figure.add_trace(go.Scatter(
            name='Pronóstico antecedente GEOGloWS',
            x=geoglows_records.index,
            y=geoglows_records.iloc[:, 0].values,
            line=dict(color='#FFA15A')))
        x_vals = (geoglows_records.index[0], stats_df.index[len(stats_df.index) - 1], stats_df.index[len(stats_df.index) - 1], geoglows_records.index[0])
        max_visible = max(max(geoglows_records.max()), max_visible)
    #
    sonics_records = historical_sonics.loc[historical_sonics.index >= pd.to_datetime(historical_sonics.index[-1] - dt.timedelta(days=8))]
    sonics_records = sonics_records.loc[sonics_records.index <= pd.to_datetime(historical_sonics.index[-1] + dt.timedelta(days=2))]
    #
    if len(sonics_records.index) > 0:
        hydroviewer_figure.add_trace(go.Scatter(
            name='Pronóstico antecedente SONICS',
            x=sonics_records.index,
            y=sonics_records.iloc[:, 0].values,
            line=dict(color='#FFA15A')))
        if sonics_records.index[0] < geoglows_records.index[0]:
            x_vals = (sonics_records.index[0], stats_df.index[len(stats_df.index) - 1], stats_df.index[len(stats_df.index) - 1],sonics_records.index[0])
        max_visible = max(max(sonics_records.max()), max_visible)
    #
    if len(forecast_gfs_df.index) > 0:
        hydroviewer_figure.add_trace(go.Scatter(
            name='Pronóstico GFS',
            x=forecast_gfs_df.index,
            y=forecast_gfs_df['Streamflow (m3/s)'],
            showlegend=True,
            line=dict(color='black', dash='dash')))

        max_visible = max(max(forecast_gfs_df.max()), max_visible)
    #
    if len(forecast_eta_eqm_df.index) > 0:
        hydroviewer_figure.add_trace(go.Scatter(
            name='Pronóstico ETA eqm',
            x=forecast_eta_eqm_df.index,
            y=forecast_eta_eqm_df['Streamflow (m3/s)'],
            showlegend=True,
            line=dict(color='blue', dash='dash')))
        max_visible = max(max(forecast_eta_eqm_df.max()), max_visible)
    #
    if len(forecast_eta_scal_df.index) > 0:
        hydroviewer_figure.add_trace(go.Scatter(
            name='Pronóstico ETA scal',
            x=forecast_eta_scal_df.index,
            y=forecast_eta_scal_df['Streamflow (m3/s)'],
            showlegend=True,
            line=dict(color='green', dash='dash')))
        max_visible = max(max(forecast_eta_scal_df.max()), max_visible)
    #
    if len(forecast_wrf_df.index) > 0:
        hydroviewer_figure.add_trace(go.Scatter(
            name='Pronóstico WRF',
            x=forecast_wrf_df.index,
            y=forecast_wrf_df['Streamflow (m3/s)'],
            showlegend=True,
            line=dict(color='brown', dash='dash')))
        max_visible = max(max(forecast_wrf_df.max()), max_visible)
    #
    r2_33 = round(rperiods_geoglows.iloc[0]['return_period_2_33'], 2)
    colors = { '2.33 Year': 'rgba(243, 255, 0, .4)', '5 Year': 'rgba(255, 165, 0, .4)', '10 Year': 'rgba(255, 0, 0, .4)'}
    #
    if max_visible > r2_33:
        visible = True
        hydroviewer_figure.for_each_trace(
            lambda trace: trace.update(visible=True) if trace.name == "Máximos y mínimos" else (), )
    else:
        visible = 'legendonly'
        hydroviewer_figure.for_each_trace(
            lambda trace: trace.update(visible=True) if trace.name == "Máximos y mínimos" else (), )
    #
    def template(name, y, color, fill='toself'):
        return go.Scatter(
            name=name,
            x=x_vals,
            y=y,
            legendgroup='returnperiods',
            fill=fill,
            visible=visible,
            line=dict(color=color, width=0))
    #
    r5 = round(rperiods_geoglows.iloc[0]['return_period_5'], 2)
    r10 = round(rperiods_geoglows.iloc[0]['return_period_10'], 2)
    #
    hydroviewer_figure.add_trace(template('Periodos de retorno', (r10 * 0.05, r10 * 0.05, r10 * 0.05, r10 * 0.05), 'rgba(0,0,0,0)', fill='none'))
    hydroviewer_figure.add_trace(template(f'2.33 años: {r2_33}', (r2_33, r2_33, r5, r5), colors['2.33 Year']))
    hydroviewer_figure.add_trace(template(f'5 años: {r5}', (r5, r5, r10, r10), colors['5 Year']))
    hydroviewer_figure.add_trace(template(f'10 años: {r10}', (r10, r10, max(r10 + r10 * 0.05, max_visible), max(r10 + r10 * 0.05, max_visible)), colors['10 Year']))
    #
    hydroviewer_figure['layout']['xaxis'].update(autorange=True)
    return(hydroviewer_figure)




