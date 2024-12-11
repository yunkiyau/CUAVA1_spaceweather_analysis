import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
import math
import pandas as pd
from astropy.timeseries import LombScargle
import scipy
from scipy import signal
import statsmodels
from statsmodels.tsa.tsatools import detrend
from scipy.signal import detrend
import statsmodels.api as sm

#-------------------Basic Functions------------------------------------------------

#---------Basic functions for TLE (Two-Line Element) files (satellite data)------------------
def parse_time_value(time_value): #Function to get calender date for input datetime value.
    year = int(time_value[:2]) + 2000 #extract first two digits of time_value and add 2000
    day_of_year = float(time_value[2:]) #extract everything after the first two digits
    return datetime(year, 1, 1) + timedelta(days=day_of_year) #(year, month, day)+ difference from Jan 1st. Gives the date of (jan 1st + day_of_year). 


def days_since_epoch(time_value, epoch=datetime(2021, 1, 1)): #function to compute decimal days since launch for an input datetime (Jan 1st 2021 is a placeholder for mission start date)
    year = int(time_value[:2]) + 2000
    day_of_year = float(time_value[2:])
    dt = datetime(year, 1, 1) + timedelta(days=day_of_year)
    delta = dt - epoch #difference between two dates
    return delta.total_seconds() / (60 * 60 * 24) #the above date difference in decimal days (so decimal days since launch)


def read_data(dataset): #function to extract eccentricity values and dates for input mission TLE file
    global missionstart #missionstart applies to the whole script
    for line in open(str(dataset)): #str refers to a string within the brackets that refer to the text file
        values = line.replace("  ", " ") #replace doublespace with single space
        values = line.split(" ") #splits line into a list
        values = [x for x in values if x != ''] #remove any empty elements in 'values' list

        if line.startswith("1"): 
            if missionstart == 0:
                missionstart = parse_time_value(values[3]) #missionstart gets updated with the first date of the mission. values[3] is the fourth element of values
        
            days.append(days_since_epoch(values[3], epoch = missionstart)) #add elements to the days array using the output of the days since epoch function
            days_datetime.append(parse_time_value(values[3])) #add elements to days_datetime in datetime format

        if line.startswith("2"):
            tempeccentricity = "0." + values[4]
            eccentricity.append(float(tempeccentricity)) #adds eccentricity entries to the eccentricity array
            

# function to take the time derivative of an input orbital parameter array (ie. eccentricity, raan, etc)
def time_derivative(orbital_parameter_array): #assumes you have a list of values ordered by sequential decimal day (ie. output from read_data function)
    parameter_time_derivative = [0]
    for i in range(1, len(orbital_parameter_array)):
        parameter_time_derivative.append(orbital_parameter_array[i] - orbital_parameter_array[i-1])
    return parameter_time_derivative


#-------------Basic functions for OMNI data (spaceweather data files)--------------------

def parse_timestamp(year, doy, hour): #arguments are the first 3 columns of the OMNI data. Function returns Y/M/D/H/M format for the OMNI DOY entry
    dt = datetime(year=int(year), month=1, day=1) + timedelta(days=int(doy) - 1, hours=int(hour)) 
    return dt 


def read_spaceweather(days_datetime): #function to read OMNI data for input mission date range (provided from read_data function)

    # Sort the list of datetimes in ascending order
    days_datetime.sort()

    # Read the OMNI file and extract the relevant values
    with open('CUAVA-1_Omni2.txt') as f:
        for line in f:
            values = line.split(" ")
            values = [x for x in values if x != '']
            year = int(values[0])
            doy = int(values[1])
            hour = int(values[2])
            f10_val = float(values[6])
            KP_val = int(values[3])
            DST_val = int(values[4])
            AP_val = int(values[5])

            # Parse the datetime from the file
            dt = parse_timestamp(year, doy, hour)

            # Check if the datetime is after 2018
            if dt.year < 2018:
                continue

            if days_datetime[0] <= dt <= days_datetime[-1]: #only extract spaceweather data if date is within the mission date range
                f10.append(f10_val) #add the f10 value to the f10 list
                weather_days.append((dt - days_datetime[0]).days) #add values to f10 days - the value is (dt(the DOY in datetime) minus the first day of the mission) in days 
                KPIndex.append(KP_val)
                DST.append(DST_val)
                AP.append(AP_val)

#-----------------Initiate required empty variables---------------------------------------------------

eccentricity = []
days = []
days_datetime = []
missionstart = 0
f10 = []
weather_days = []
KPIndex = []
DST = []
AP = []
raan = []

#--------------------Data Analysis Section for Cuava 1 cubesat---------------------------------------------------------

#-------Read satellite TLE data, extract spaceweather for the mission and compute time derivative of eccentricity-----
read_data('cuava1.txt') #apply the read function to cuava1 TLE
read_spaceweather(days_datetime)  #extract OMNI data for the mission duration
dedt = time_derivative(eccentricity) #create eccentricity derivative array


#-----------Denoising data using Savitsky-Golay Filter-------------------------------

e_smooth = signal.savgol_filter(eccentricity, window_length = 200, polyorder = 3, mode = "interp")
#plt.plot(days, eccentricity, 'b-', linewidth = 0.5)
#plt.xlabel('Decimal Days since launch', fontsize = 23)
#plt.ylabel('Eccentricity', color = 'blue', fontsize = 23)
#plt.xticks(fontsize=23)  
#plt.yticks(fontsize=23)

dedt_smooth = signal.savgol_filter(dedt, window_length = 200, polyorder = 3, mode = "interp")
#plt.plot(days, dedt, 'b-', linewidth = 0.5)
#plt.xlabel('Decimal Days since launch', fontsize = 23)
#plt.ylabel('de/dt', color = 'blue', fontsize = 23)
#plt.xticks(fontsize=23)  
#plt.yticks(fontsize=23)

f10_smooth = signal.savgol_filter(f10, window_length = 200, polyorder = 3, mode = "interp")
dst_smooth = signal.savgol_filter(DST, window_length = 200, polyorder = 3, mode = "interp")
ap_smooth = signal.savgol_filter(AP, window_length = 200, polyorder = 3, mode = "interp")

#-------------------Plotting satellite and spaceweather data on the same graph----------------
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# plot first data set on first axis
ax1.plot(days, dedt, 'b-', linewidth = 1, label='de/dt')
ax1.set_xlabel('Decimal Days since launch', fontsize = 18)
ax1.set_ylabel('de/dt', color = 'blue', fontsize = 18)

# plot second data set on second axis
ax2.plot(weather_days, f10, 'r-', linewidth = 1, label='F10.7cm solar radio flux')
ax2.set_ylabel('F10.7cm solar radio flux (sfu)', color = 'red', fontsize = 18)


# add legends
ax1.legend(loc='upper left', fontsize = 15)
ax2.legend(loc='upper right', fontsize = 15)

plt.show()

#------------------------Lomb_Scargle Periodogram Section-------------------------------------------------------

#--------detrending data subsection-----------------------------
min_freq = 0.001
max_freq = 0.08
sampling_freq = 100


n = len(f10) #change to desired weather index for polynomial detrend
t = range(n)
p = np.polyfit(t,f10,2) #change desired weather index for polynomial detrend
y = np.polyval(p,t)

n2 = len(dedt) #change to desired orbital parameter for polynomial detrend
t2 = range(n2)
p2 = np.polyfit(t2,dedt,3) #change desired orbital parameter for polynomial detrend
y2 = np.polyval(p2,t2)

e_detrended = detrend(eccentricity, type = 'constant')
dedt_detrended = dedt - y2
KP_detrended = detrend(KPIndex, type = 'constant')
f10_detrended = f10 - y
DST_detrended = DST - y
ap_detrended = detrend(AP, type = 'constant')

#------------------------Periodograms on detrended data-------------------------------
freq_e, p_e = LombScargle(days, e_detrended).autopower(minimum_frequency=min_freq, maximum_frequency=max_freq, samples_per_peak=sampling_freq) 

freq_KP, p_KP = LombScargle(weather_days, KP_detrended, fit_mean=True, center_data=True).autopower(minimum_frequency=min_freq, maximum_frequency=max_freq, samples_per_peak=sampling_freq)

freq_f10, p_f10 = LombScargle(weather_days, f10_detrended).autopower(minimum_frequency=min_freq, maximum_frequency=max_freq, samples_per_peak=sampling_freq)

freq_DST, p_DST = LombScargle(weather_days, DST_detrended, fit_mean=True, center_data=True).autopower(minimum_frequency=min_freq, maximum_frequency=max_freq, samples_per_peak=sampling_freq)

freq_ap, p_ap = LombScargle(weather_days, ap_detrended, fit_mean=True, center_data=True).autopower(minimum_frequency=min_freq, maximum_frequency=max_freq, samples_per_peak=sampling_freq)


# Create a figure with two subplots
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(freq_e, p_e, 'b-', linewidth = 0.5, label='eccentricity')
ax1.set_xlabel('Frequency/day')
ax1.set_ylabel('Power (eccentricity)', color='blue')

ax2.plot(freq_ap, p_ap, 'r-', linewidth = 0.3, label='Ap Index')
ax2.set_xlabel('Frequency/day')
ax2.set_ylabel('Power (Ap Index)', color='red')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')



#----------------------Uncertainties for LSPs (calculating 1/2 FWHM of highest peak)--------------------------------------------------
max_power = max(abs(p_e)) #change desired measurement here (use the power spectral density from the LSP of the orbital parameter)
max_power_index = np.argmax(abs(p_e)) #change desired measurement here
halfmaxpower = abs(max_power / 2)


#-------------Functions to approach the peak from the left and right to find the values on the curve at 1/2 max------------
def parse_array_leftwards(array, start_index, target_value):
    index = start_index
    rounded_target = round(target_value, 5)
    while index >= 0:
        rounded_element = round(array[index],5)
        if rounded_element <= rounded_target:
            break
        else:
            index = index - 1
    return index

def parse_array_rightwards(array, start_index, target_value):
    index = start_index
    rounded_target = round(target_value, 5)
    while index < len(array):
        rounded_element = round(array[index],5)
        if rounded_element <= rounded_target:
            break
        else:
            index = index + 1
    return index

LSPleft_index = parse_array_leftwards(p_e, max_power_index, halfmaxpower) 
LSPright_index = parse_array_rightwards(p_e, max_power_index, halfmaxpower) 
leftfreq = freq_e[LSPleft_index] 
rightfreq = freq_e[LSPright_index] 
LSPhalfhalffwhm = (abs(rightfreq - leftfreq))/2

freq = freq_e[max_power_index] #get the frequency of the highest peak

print("max_power:", max_power)
print("corresponding freq:", freq)
print("Freq uncertainty (1/2 FWHM):", LSPhalfhalffwhm)

#------------------------Cross-Correlation Section-------------------------------------------------------

#Creating a table of daily average OMNI values
omnitable = pd.DataFrame({'decimaldays':weather_days, 'f10':f10, 'kp':KPIndex, 'dst':DST, 'ap':AP})
omnitable['wholenumberdays'] = pd.cut(omnitable['decimaldays'], bins=328, labels=False) #change bins to total mission days
dailyf10 = omnitable.groupby('wholenumberdays')['f10'].agg('mean').values
dailykp = omnitable.groupby('wholenumberdays')['kp'].agg('mean').values
dailydst = omnitable.groupby('wholenumberdays')['dst'].agg('mean').values
dailyap = omnitable.groupby('wholenumberdays')['ap'].agg('mean').values

#Creating a table of daily average TLE orbital parameters
tletable = pd.DataFrame({'decimaldays':days, 'e':eccentricity, 'dedt':dedt})
tletable['wholenumberdays'] = pd.cut(tletable['decimaldays'], bins=328, labels=False) #change bins to total mission days
dailye = tletable.groupby('wholenumberdays')['e'].agg('mean').values
dailydedt = tletable.groupby('wholenumberdays')['dedt'].agg('mean').values

#Performing the Cross-Correlation
forwards = sm.tsa.stattools.ccf(dailye, dailyf10, adjusted=False)
backwards = sm.tsa.stattools.ccf(dailye[::-1], dailyf10[::-1], adjusted=False)[::-1]
corr = np.concatenate([backwards, forwards[1:]])


full = pd.DataFrame({'Lag (days)':np.arange(len(corr))-len(backwards), 
                     'CC coefficient':corr})

full.plot(x='Lag (days)', linewidth = 0.8)
plt.ylabel('Cross-correlation coefficient (vs eccentricity)')
plt.title('f10')


#---------------------Uncertainties for X-Correlations--------------------------------------------
# Uncertainties for coefficients are their SDs, uncertainties for the lags are their FWHM
lagdays = full['Lag (days)'].values
corcfs = full['CC coefficient'].values
max_corr = max(abs(corcfs))
max_corr_index = np.argmax(abs(corcfs))
halfmaxcorr = abs(max_corr / 2)

#----Calculating the FWHM for lags--------------------------
left_index = parse_array_leftwards(corcfs, max_corr_index, halfmaxcorr)
right_index = parse_array_rightwards(corcfs, max_corr_index, halfmaxcorr)
leftlag = lagdays[left_index]
rightlag = lagdays[right_index]
halffwhm = (abs(rightlag - leftlag))
lag = lagdays[max_corr_index]
stcorr = np.std(corr)

print("(abs) max coefficient value:", max_corr)
print("Standard deviation of coefficients:", stcorr)
print("Corresponding lag:", lag)
print("Lag uncertainty (fwhm):", halffwhm)

#----------------------------------------------------------------------------------------------------

#plt.show()
#plt.show()


