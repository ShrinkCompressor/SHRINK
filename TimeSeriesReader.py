import numpy as np
import gzip
from TimeSeries import TimeSeries
from Point import Point
import numpy as np
import csv
import os


class TimeSeriesReader:
    @staticmethod
    def getTimeSeries(csv_file):
        ts = []
        max_val = float("-inf")
        min_val = float("inf")

        try:
            with open(csv_file,'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    timestamp = int(row[0])
                    value = float(row[1])
                    ts.append(Point(timestamp, value))
                    max_val = max(max_val, value)
                    min_val = min(min_val, value)
        except Exception as e:
            print(e)
        ts = TimeSeries(ts, max_val - min_val)
        # ts.size = os.path.getsize(csv_file)###去掉这个为默认npy的大小，本方法使用csv

        return ts


# class TimeSeriesReader:
#     @staticmethod
#     def getTimeSeries(inputStream, delimiter):
#         max_val = float("-inf")
#         min_val = float("inf")

#         # Initialize arrays to store timestamps and values
#         timestamps = []
#         values = []

#         try:
#             # Load data directly into NumPy arrays
#             data = np.loadtxt(inputStream, delimiter=delimiter)
#             timestamps = data[:, 0].astype(np.int32)
#             values = data[:, 1].astype(np.float32)

#             # Calculate min and max values
#             max_val = np.max(values)
#             min_val = np.min(values)

#         except Exception as e:
#             print(e)

#         # Create TimeSeries object
#         ts = [Point(timestamp, val) for timestamp, val in zip(timestamps, values)]
#         return TimeSeries(ts, max_val - min_val)


# class TimeSeriesReader:
#     @staticmethod
#     def getTimeSeries(inputStream, delimiter, gzipFlag):
#         max_val = float("-inf")
#         min_val = float("inf")

#         # Initialize arrays to store timestamps and values
#         timestamps = []
#         values = []

#         try:
#             with gzip.open(inputStream, 'rt', encoding='utf-8') as file:
#                 # Load data directly into NumPy arrays
#                 data = np.loadtxt(file, delimiter=delimiter)
#                 timestamps = data[:, 0].astype(int)
#                 values = data[:, 1].astype(float)

#             # Calculate min and max values
#             max_val = np.max(values)
#             min_val = np.min(values)

#         except Exception as e:
#             print(e)

#         # Create TimeSeries object
#         ts = [Point(timestamp, val) for timestamp, val in zip(timestamps, values)]
#         return TimeSeries(ts, max_val - min_val)




# import sys
# import gzip
# from typing import List
# from io import BufferedReader, BytesIO
# from TimeSeries import TimeSeries
# from Point import Point

# class TimeSeriesReader:
#     @staticmethod
#     def getTimeSeries(inputStream, delimiter, gzipFlag):
#         ts = []
#         max_val = float("-inf")
#         min_val = float("inf")

#         try:
#             if gzipFlag:
#                 with gzip.open(inputStream, 'rt', encoding='utf-8') as file:
#                     for line in file:
#                         elements = line.strip().split(delimiter)
#                         timestamp = int(elements[0])
#                         value = float(elements[1])
#                         ts.append(Point(timestamp, value))

#                         max_val = max(max_val, value)
#                         min_val = min(min_val, value)
#             else:
#                     with open(inputStream, 'r', encoding='utf-8') as file:
#                         print("inputStream = ", inputStream)
#                         for line in file:
#                             elements = line.strip().split(delimiter)
#                             timestamp = int(elements[0])
#                             value = float(elements[1])
#                             print(value)
#                             ts.append(Point(timestamp, value))

#                             max_val = max(max_val, value)
#                             min_val = min(min_val, value)
            

#             '''
#             decoder = inputStream.read().decode('utf-8')
#             bufferedReader = BufferedReader(BytesIO(decoder))

#             for line in bufferedReader:
#                 elements = line.split(delimiter)
#                 timestamp = int(elements[0])
#                 value = float(elements[[1]])
#                 ts.append(Point(timestamp, value))

#                 max_val = max(max_val, value)
#                 min_val = min(min_val, value)
#             '''
#         except Exception as e:
#             print(e)

#         return TimeSeries(ts, max_val - min_val)
