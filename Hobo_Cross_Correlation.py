import pandas as pd
import os
from scipy.signal import correlate
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from math import sqrt
import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
# For Normalization of time series I used the following examples
# https://machinelearningmastery.com/normalize-standardize-time-series-data-python/
# https://machinelearningmastery.com/load-explore-time-series-data-python/

def resample_and_normalize_series(series, resample):
    Series = series.resample(resample)
    upsampled = Series.interpolate(method='linear')

    # prepare data for standardization
    values = upsampled.values
    values = values.reshape((len(values), 1))
    # train the standardization
    scaler = StandardScaler()
    scaler = scaler.fit(values)
    print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
    # standardization the dataset and print the first 5 rows
    return scaler.transform(values)

def getDateRange(series, start_date, end_date, freq):
    return series[series.index.isin(pd.date_range(start_date, end_date, freq=freq))]

def cross_correlation_using_fft(x, y):
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)

os.chdir(r'C:\Users\rmcd\Documents\GreenRiverLarval\Hobo_GPS_Locations')

Jenson_Q = pd.Series.from_csv('Jenson_Q.csv', header=0, index_col='Date', parse_dates=True)
Jenson_Q_HF = getDateRange(Jenson_Q, "2018-03-27 12:00:00", "2018-03-28 12:00:00", 'T')
nJenson_Q = resample_and_normalize_series(Jenson_Q, '15T')
nJenson_Q_HF = resample_and_normalize_series(Jenson_Q_HF, '15T')
Jenson_Q_LF = getDateRange(Jenson_Q, "2018-03-06 00:00:00", "2018-03-08 00:00:00", 'T')
nJenson_Q_LF = resample_and_normalize_series(Jenson_Q_LF, '15T')

Hobo_6_I = pd.Series.from_csv('Hobo_6_I.csv',  header=0, index_col='Date', parse_dates=True)
Hobo_6_I_HF = getDateRange(Hobo_6_I, "2018-03-27 12:00:00", "2018-03-28 12:00:00", 'T')
nHobo_6_I = resample_and_normalize_series(Hobo_6_I, '15T')
nHobo_6_I_HF = resample_and_normalize_series(Hobo_6_I_HF, '15T')

Hobo_4 = pd.Series.from_csv('Hobo_4.csv',  header=0, index_col='Date', parse_dates=True)
Hobo_4_HF = getDateRange(Hobo_4, "2018-03-27 12:00:00", "2018-03-28 12:00:00", 'T')
nHobo_4 = resample_and_normalize_series(Hobo_4, '15T')
nHobo_4_HF = resample_and_normalize_series(Hobo_4_HF, '15T')

Hobo_1 = pd.Series.from_csv('Hobo_1.csv',  header=0, index_col='Date', parse_dates=True)
Hobo_1_HF = getDateRange(Hobo_1, "2018-03-27 12:00:00", "2018-03-28 12:00:00", 'T')
nHobo_1 = resample_and_normalize_series(Hobo_1, '15T')
nHobo_1_HF = resample_and_normalize_series(Hobo_1_HF, '15T')

Hobo_SLE1 = pd.Series.from_csv('Hobo_SLE1.csv',  header=0, index_col='Date', parse_dates=True)
Hobo_SLE1_HF = getDateRange(Hobo_SLE1, "2018-03-27 12:00:00", "2018-03-28 12:00:00", 'T')
nHobo_SLE1 = resample_and_normalize_series(Hobo_SLE1, '15T')
nHobo_SLE1_HF = resample_and_normalize_series(Hobo_1_HF, '15T')
Hobo_SLE1_LF = getDateRange(Hobo_SLE1, "2018-03-06 00:00:00", "2018-03-08 00:00:00", 'T')
nHobo_SLE1_LF = resample_and_normalize_series(Hobo_SLE1_LF, '15T')

plt.plot(nJenson_Q)
plt.plot(nHobo_6_I)
plt.plot(nHobo_4)
plt.plot(nHobo_1)
plt.plot(nHobo_SLE1)
plt.show()
#
plt.plot(nJenson_Q_HF)
plt.plot(nJenson_Q_LF)
# plt.plot(nHobo_6_I_HF)
# plt.plot(nHobo_4_HF)
# plt.plot(nHobo_1_HF)
plt.plot(nHobo_SLE1_HF)
plt.plot(nHobo_SLE1_LF)
plt.show()

corrHobo_sle1 = correlate(nJenson_Q_HF, nHobo_SLE1_HF)
zero_index = int(len(nJenson_Q_HF) / 2) - 1
shift = zero_index - np.argmax(corrHobo_sle1)
print(shift)
# shift is 27 or 405 minutes or 6.75 hours or 6 hours 45 minutes for high flow
corrHobo_sle2 = correlate(nJenson_Q_LF, nHobo_SLE1_LF)
zero_index = int(len(nJenson_Q_LF) / 2) - 1
shift = zero_index - np.argmax(corrHobo_sle2)
print(shift)
# shift is 61 or 915 minutes or 15.25 hours or 15 hours 15 minutes for high flow

JensonQ = getDateRange(Jenson_Q, "2018-03-05 00:00:00", "2018-03-06 00:00:00", 'T')
np.append(JensonQ, Jenson_Q_HF)
HoboSLE1 = getDateRange(Hobo_SLE1, "2018-03-05 10:00:00", "2018-03-06 10:00:00", 'T')
np.append(HoboSLE1, getDateRange(Hobo_SLE1, "2018-03-27 18:45:00", "2018-03-28 18:45:00", 'T'))

# plt.plot(JensonQ, HoboSLE1)
plt.plot(getDateRange(Jenson_Q, "2018-03-05 00:00:00", "2018-03-06 00:00:00", 'T'),
         getDateRange(Hobo_SLE1, "2018-03-05 10:00:00", "2018-03-06 10:00:00", 'T'))
plt.plot(Jenson_Q_HF, getDateRange(Hobo_SLE1, "2018-03-27 18:45:00", "2018-03-28 18:45:00", 'T'))
plt.show()
jq = getDateRange(Jenson_Q, "2018-03-05 00:00:00", "2018-03-06 00:00:00", 'T')
print(type(jq))
jq1 = getDateRange(Jenson_Q, "2018-03-27 12:00:00", "2018-03-28 12:00:00", 'T')
jqq = pd.concat([jq, jq1])
sle1 = getDateRange(Hobo_SLE1, "2018-03-05 10:00:00", "2018-03-06 10:00:00", 'T')
sle1a = getDateRange(Hobo_SLE1, "2018-03-27 18:45:00", "2018-03-28 18:45:00", 'T')
sle1b = pd.concat([sle1,sle1a])
df = pd.DataFrame({"Jensen Discharge" : jqq.values.transpose(), "HOBO_SLE1" : sle1b.values.transpose()})
df.to_csv("HoboSLE1Stage_Discharge.csv", index=False)