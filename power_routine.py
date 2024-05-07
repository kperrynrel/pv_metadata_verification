# Import the associated packages
import pvanalytics
import rdtools
import pvlib
import pandas as pd
from pvanalytics.quality import data_shifts as ds
from pvanalytics.quality import gaps
from pvanalytics.quality.outliers import zscore
from pvanalytics.features.daytime import power_or_irradiance
from pvanalytics.quality.time import shifts_ruptures
from pvanalytics.features import daytime
from pvanalytics.system import (is_tracking_envelope,
                                infer_orientation_fit_pvwatts)
from statistics import mode
import ruptures as rpt
import pvlib
import time

nsrdb_api_key = '4z5fRAXbGB3qldVVd3c6WH5CuhtY5mhgC2DyD952'
nsrdb_user_email = "kirsten.perry@nrel.gov"

def pull_power_data(system_id, grouping, timezone):
    """
    Pull the AC power data associated with the site.
    """
    try:
        pv_dataframe = pd.read_csv("C:/tmp/PVFleetDataInitiative/Results/" +
                                   grouping + "/data/C" + str(system_id) + ".csv",
                                   index_col=0, parse_dates=True)
        pv_dataframe.index = pv_dataframe.index.tz_localize("UTC",
                                                            ambiguous=True,
                                                            nonexistent='NaT')
        if timezone == '4':
            timezone = 'Etc/GMT+4'
        elif timezone == '5':
            timezone = 'Etc/GMT+5'
        elif timezone == '6':
            timezone = 'Etc/GMT+6'
        elif timezone == '7':
            timezone = 'Etc/GMT+7'
        elif timezone == '8':
            timezone = 'Etc/GMT+8'
        elif timezone == '9':
            timezone = 'Etc/GMT+9'
        elif timezone == '10':
            timezone = 'Etc/GMT+10'
        pv_dataframe.index = pd.to_datetime(pv_dataframe.index).tz_convert(timezone)
        power_cols = [x for x in list(pv_dataframe.columns) if 'ac_power_inv' in x]
        pv_dataframe = pv_dataframe[power_cols]
        # Remove any duplicate index values (this is just a failsafe in case there are!)
        pv_dataframe = pv_dataframe.groupby(pv_dataframe.index).first()
    except:
        print("System failure: " + str(system_id))
        pv_dataframe = pd.DataFrame()
    return pv_dataframe

def pull_nsrdb_data(time_series, system_id, grouping):
    """
    Pull the NSRDB data for a particular system.
    """
    # Get the time frequency of the time series
    time_series.index = pd.to_datetime(time_series.index)
    freq_minutes = mode(
        time_series.index.to_series().diff(
        ).dt.seconds / 60)
    data_freq = str(freq_minutes) + "min"
    psm3s = []
    years = list(time_series.index.year.drop_duplicates())
    years = [int(x) for x in years if x <= 2020]
    for year in years:
        psm3 = pd.read_csv("C:/tmp/PVFleetDataInitiative/Results/" + grouping
                           + "/nsrdb/" + str(system_id) + "_"+ str(year) + ".csv",
                           index_col=0, parse_dates=True)
        psm3s.append(psm3)
    if len(psm3s) > 0:
        psm3 = pd.concat(psm3s)
        psm3 = psm3.groupby(psm3.index).first()
        psm3 = psm3.reindex(pd.date_range(psm3.index[0],
                                  psm3.index[-1],
                                  freq=data_freq)).interpolate()
        psm3.index = psm3.index.tz_convert(time_series.index.tz)
        psm3 = psm3.reindex(time_series.index).interpolate()
    return psm3

def pvanalytics_mount_check(estimated_az, estimated_tilt, estimated_mount,
                            gt_az, gt_tilt, gt_mount, system_id,
                            az_diff_degrees_max = 15,
                            tilt_diff_degrees_max = 5):
    """
    PVAnalytics checks for mounting data.
    """
    mount_match_found, az_tilt_match_found = False, False
    # PVANALYTICS CHECKS
    abs_az_diff = abs(int(estimated_az - gt_az))
    abs_tilt_diff = abs(int(estimated_tilt - gt_tilt))
    if ((abs_az_diff <= az_diff_degrees_max) &
        (abs_tilt_diff <= tilt_diff_degrees_max)):
        print("PVAnalytics az-tilt match found for system " + str(system_id))
        az_tilt_match_found = True
    if ((estimated_mount == 'FIXED') & (gt_mount == False)):
        print("PVAnalytics mount match found for system " + str(system_id))
        mount_match_found = True
    if ((estimated_mount == 'TRACKING') & (gt_mount == True)):
        print("PVAnalytics mount match found for system " + str(system_id))
        mount_match_found = True 
    return az_tilt_match_found, mount_match_found


def panel_segmentation_mount_check(estimated_az, estimated_mount,
                                   gt_az, gt_mount, system_id,
                                   az_diff_degrees_max = 15):
    """
    PVAnalytics checks for mounting data.
    """
    mount_match_found, az_match_found = False, False
    # PVANALYTICS CHECKS
    abs_az_diff = abs(int(estimated_az - gt_az))
    if (abs_az_diff <= az_diff_degrees_max):
        print("Panel-Segmentation az match found for system " + str(system_id))
        az_match_found = True
    if (('fixed' in estimated_mount) & (gt_mount == False)):
        print("Panel-Segmentation mount match found for system " + str(system_id))
        mount_match_found = True
    if (('track' in estimated_mount) & (gt_mount == True)):
        print("Panel-Segmentation mount match found for system " + str(system_id))
        mount_match_found = True 
    return az_match_found, mount_match_found


def run_power_stream_routine(power_time_series, latitude, longitude, psm3):
    """
    Function stringing all the PVAnalytics functions together, where
    az/tilt and mounting config are estimated.
    """
    # Get the time frequency of the time series
    freq_minutes = mode(
        power_time_series.index.to_series().diff(
        ).dt.seconds / 60)
    data_freq = str(freq_minutes) + "min"
    power_time_series = power_time_series.asfreq(data_freq)
    
    # BASIC DATA CHECKS (STALE, OUTLIERS)
    
    # REMOVE STALE DATA (that isn't during nighttime periods or clipped)
    # Day/night mask
    daytime_mask = power_or_irradiance(power_time_series)
    # Clipped data (uses rdtools filters)
    clipping_mask = rdtools.filtering.xgboost_clip_filter(
        power_time_series)
    stale_data_mask = gaps.stale_values_round(
        power_time_series,
        window=3,
        decimals=2)
    
    stale_data_mask = (stale_data_mask & daytime_mask
                       & clipping_mask)
    # REMOVE NEGATIVE DATA
    negative_mask = (power_time_series < 0)
    # FIND ABNORMAL PERIODS
    daily_min = power_time_series.resample('D').min()   
    series_min = 0.1 * power_time_series.mean()
    erroneous_mask = (daily_min >= series_min)
    erroneous_mask = erroneous_mask.reindex(
        index=power_time_series.index,
        method='ffill',
        fill_value=False)
    # FIND OUTLIERS (Z-SCORE FILTER)
    zscore_outlier_mask = zscore(power_time_series,
                                 zmax=4,
                                 nan_policy='omit')
    # Filter the time series, taking out all of the issues
    issue_mask = ((~stale_data_mask) & (~negative_mask) &
              (~erroneous_mask) & (~zscore_outlier_mask))
    
    power_time_series = power_time_series[issue_mask].copy()
    power_time_series = power_time_series.asfreq(data_freq)
    
    # DATA COMPLETENESS CHECK
    
    # Visualize daily data completeness
    power_time_series = power_time_series.asfreq(data_freq)
    daytime_mask = power_or_irradiance(power_time_series)
    power_time_series.loc[~daytime_mask] = 0
    # Trim the series based on daily completeness score
    trim_series_mask = \
    pvanalytics.quality.gaps.trim_incomplete(
        power_time_series,
        minimum_completeness=.25,
        freq=data_freq)
    
    power_time_series = power_time_series[trim_series_mask]
    
    # TIME SHIFT DETECTION
    if ((len(power_time_series.resample('D').mean().dropna()) >=150) &
            (len(power_time_series.drop_duplicates()) > 1000)):
        # Get the modeled sunrise and sunset time series based
        # on the system's latitude-longitude coordinates
        modeled_sunrise_sunset_df = \
        pvlib.solarposition.sun_rise_set_transit_spa(
             power_time_series.index, latitude, longitude)
        
        # Calculate the midday point between sunrise and sunset
        # for each day in the modeled power series
        modeled_midday_series = \
        modeled_sunrise_sunset_df['sunrise'] + \
            (modeled_sunrise_sunset_df['sunset'] -
             modeled_sunrise_sunset_df['sunrise']) / 2
        
        # Run day-night mask on the power time series
        daytime_mask = power_or_irradiance(
            power_time_series, freq=data_freq,
            low_value_threshold=.005)
        
        # Generate the sunrise, sunset, and halfway points
        # for the data stream
        sunrise_series = daytime.get_sunrise(daytime_mask)
        sunset_series = daytime.get_sunset(daytime_mask)
        midday_series = sunrise_series + ((sunset_series -
                                           sunrise_series)/2)
        
        # Convert the midday and modeled midday series to daily
        # values
        midday_series_daily, modeled_midday_series_daily = (
            midday_series.resample('D').mean(),
            modeled_midday_series.resample('D').mean())
        
        # Set midday value series as minutes since midnight,
        # from midday datetime values
        midday_series_daily = (
            midday_series_daily.dt.hour * 60 +
            midday_series_daily.dt.minute +
            midday_series_daily.dt.second / 60)
        modeled_midday_series_daily = \
            (modeled_midday_series_daily.dt.hour * 60 +
             modeled_midday_series_daily.dt.minute +
             modeled_midday_series_daily.dt.second / 60)
        
        # Estimate the time shifts by comparing the modelled
        # midday point to the measured midday point.
        is_shifted, time_shift_series = shifts_ruptures(
            midday_series_daily, modeled_midday_series_daily,
            period_min=15, shift_min=15, zscore_cutoff=1.5)
        
        # Build a list of time shifts for re-indexing. We choose to use dicts.
        time_shift_series.index = pd.to_datetime(
            time_shift_series.index)
        changepoints = (time_shift_series != time_shift_series.shift(1))
        changepoints = changepoints[changepoints].index
        changepoint_amts = pd.Series(time_shift_series.loc[changepoints])
        time_shift_list = list()
        for idx in range(len(changepoint_amts)):
            if changepoint_amts[idx] == 0:
                change_amt = 0
            else:
                change_amt = -1 * changepoint_amts[idx]
            if idx < (len(changepoint_amts) - 1):
                time_shift_list.append({"datetime_start":
                                        str(changepoint_amts.index[idx]),
                                        "datetime_end":
                                            str(changepoint_amts.index[idx + 1]),
                                        "time_shift": change_amt})
            else:
                time_shift_list.append({"datetime_start":
                                        str(changepoint_amts.index[idx]),
                                        "datetime_end":
                                            str(time_shift_series.index.max()),
                                        "time_shift": change_amt})
        
        # Correct any time shifts in the time series
        new_index = pd.Series(power_time_series.index, index=power_time_series.index).dropna()
        for i in time_shift_list:
            if pd.notna(i['time_shift']):
                new_index[(power_time_series.index >= pd.to_datetime(i['datetime_start'])) &
                      (power_time_series.index < pd.to_datetime(i['datetime_end']))] = \
                power_time_series.index + pd.Timedelta(minutes=i['time_shift'])
        power_time_series.index = new_index
        
        # Remove duplicated indices and sort the time series (just in case)
        power_time_series = power_time_series[~power_time_series.index.duplicated(
            keep='first')].sort_index()
                
        # DATA SHIFT DETECTION
        # Set all values in the nighttime mask to 0
        power_time_series = power_time_series.asfreq(data_freq)
        daytime_mask = power_or_irradiance(power_time_series)
        power_time_series.loc[~daytime_mask] = 0
        # Resample the time series to daily mean
        power_time_series_daily = power_time_series.resample(
            'D').mean()
        data_shift_start_date, data_shift_end_date = \
        ds.get_longest_shift_segment_dates(
            power_time_series_daily,
            use_default_models=True)
            
        power_time_series = power_time_series[
                (power_time_series.index >=
                 data_shift_start_date.tz_convert(
                     power_time_series.index.tz)) &
                (power_time_series.index <=
                 data_shift_end_date.tz_convert(
                     power_time_series.index.tz))]
        
        power_time_series = power_time_series.asfreq(data_freq)
        
        # CLIPPING DETECTION
        clipping_mask_logic = ~rdtools.filtering.logic_clip_filter(power_time_series)
        clipping_mask_xgb = ~rdtools.filtering.xgboost_clip_filter(power_time_series)
        clipping_mask = (clipping_mask_xgb | clipping_mask_logic)
        time_series = power_time_series[~clipping_mask]
        time_series = power_time_series.asfreq(data_freq)
            
        # CHECK MOUNTING CONFIGURATION
        daytime_mask = power_or_irradiance(power_time_series)
        clipping_mask = ~rdtools.filtering.xgboost_clip_filter(power_time_series)
        predicted_mounting_config = is_tracking_envelope(power_time_series,
                                                         daytime_mask,
                                                         clipping_mask)
        
        # AZ-TILT DETECTION
        # Read in associated NSRDB data for the site.
        power_time_series = power_time_series.dropna()
        psm3 = psm3.reindex(power_time_series.index)
        
        is_clear = (psm3.ghi_clear == psm3.ghi)
        is_daytime = (psm3.ghi > 0)
        time_series_clearsky = power_time_series[is_clear &
                                                 is_daytime]
        time_series_clearsky = time_series_clearsky.dropna()
        psm3_clearsky = psm3.loc[time_series_clearsky.index]
        
        # Get solar azimuth and zenith from pvlib, based on
        # lat-long coords
        solpos_clearsky = pvlib.solarposition.get_solarposition(
            time_series_clearsky.index, latitude, longitude)
        predicted_tilt, predicted_azimuth, r2 = \
            infer_orientation_fit_pvwatts(
                time_series_clearsky,
                psm3_clearsky.ghi_clear,
                psm3_clearsky.dhi_clear,
                psm3_clearsky.dni_clear,
                solpos_clearsky.zenith,
                solpos_clearsky.azimuth,
                temperature=psm3_clearsky.temp_air,
            )
        return predicted_mounting_config.name, (predicted_azimuth, predicted_tilt)
    else:
        return None, (None, None)