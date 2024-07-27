from typing import List
from Shrink import *
from Point import Point
from TimeSeriesReader import TimeSeriesReader
from datetime import datetime, timedelta
from utilFunction import *
import gzip
import sys
import os
import time
import csv
import unittest
sys.path.append('/home/guoyou/Lossless')
import QuanTRC
import sys
sys.path.append('/home/guoyou/ExtractSemantic/Data/')
path = '/home/guoyou/ExtractSemantic/Data/'



class TestSHRINK(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """
        The initial function of the class
        
        Parameters:
        """
        super().__init__(*args, **kwargs)
        self.duration = 0
        self.tsDecompressed = None
        self.decompBaseTime = 0
        self.decompResTime = 0
        self.BaseEpsilon = 0.5

    def Assert(self, shrink, epsilonPct, ts):
        self.tsDecompressed, self.decompBaseTime  = shrink.decompress()
        Dequant_val, self.decompResTime  = shrink.residualDecode(outputPath='/home/guoyou/ExtractSemantic/residuals', epsilon = epsilonPct)
        idx = 0
        for expected in self.tsDecompressed:
            actual = ts.data[idx]
            approximateVal = expected.value + Dequant_val[idx]
            if expected.timestamp != actual.timestamp:
                continue
            idx += 1
            if(epsilonPct==0):
                self.assertAlmostEqual(actual.value, approximateVal, delta=1e-7, msg="Value did not match for timestamp " + str(actual.timestamp))
            else:
                self.assertAlmostEqual(actual.value, approximateVal, delta=epsilonPct, msg="Value did not match for timestamp " + str(actual.timestamp))
        self.assertEqual(idx, len(ts.data))

    
    def run(self, filenames: List[str], epsilons) -> None:
        """
        The entrance function to extact base and residuals for many files
        
        Parameters:
        - filenames: list of the files
        - epsilonStart: the epsilon at beginning
        - epsilonStep: step for change epsilon
        - epsilonEnd: the epsilon at last
        """
        print(f"Shrink: BaseEpsilon = {self.BaseEpsilon}")
        for filename in filenames:
            ts = TimeSeriesReader.getTimeSeries(path+filename)
            ts.size = os.path.getsize(path+filename)
            print(f"{filename}: {ts.size/1024/1024:.2f}MB")
            
                
            start_time = time.time()
            shrink = Shrink(points=ts.data, epsilon=self.BaseEpsilon)
            end_time = time.time()
            baseTime = int((end_time - start_time) * 1000)
            
            binary = shrink.toByteArray(variableByte=False, zstd=False)
            origibaseSize = shrink.saveByte(binary, filename)
            inpath = '/home/guoyou/ExtractSemantic/Base/'+filename[:-7]+"_Base.bin"
            outputPath = '/home/guoyou/ExtractSemantic/Base/'
            QuanTRC.compress(inpath,outputPath)
            baseSize = os.path.getsize('/home/guoyou/ExtractSemantic/Base/codes.rc')
            # baseSize = origibaseSize
            residuals = shrink.getResiduals()

            meanCR, meanTime = 0, baseTime
            meanDec = 0
            meanResCR = 0
            decBase, decBasetime = False, 0
            for epsilonPct in epsilons:
                if (epsilonPct>=self.BaseEpsilon):
                    print(f"Epsilon: {epsilonPct }\tCompression Ratio: {ts.size/baseSize :.5f}\t Residual CR: {0}\tCompress Time: {baseTime}ms\t Decompress Time: {decBasetime} + {self.decompResTime} = {self.decompBaseTime +self.decompResTime}ms  \tRange: {ts.range :.3f}")
                    print(f"baseSize: {baseSize/1024 :.3f}KB \t Size of residual: {0}KB \t origibaseSize: {origibaseSize/1024}KB")
                    meanCR += baseSize/ ts.size
                    meanResCR += 0
                    meanTime += 0
                    meanDec += 0
                    continue

                start_time = time.time()
                residualSize = shrink.residualEncode(residuals, epsilonPct)
                end_time = time.time()
                residualTime = int((end_time - start_time) * 1000)

                compressedSize = baseSize + residualSize

                ResidualCR = ts.size/residualSize
                # CR =  compressedSize/ts.size 
                CR =  ts.size/ compressedSize

                if(decBase==False):
                    self.Assert(shrink, epsilonPct, ts) ### To assert error is bounded
                    decBase = True
                    decBasetime = self.decompBaseTime
                print(f"Epsilon: {epsilonPct }\tCompression Ratio: {CR:.5f} \t baseSize: {baseSize/1024 :.3f}KB \t residualSize: {residualSize/1024 :.3f}KB \tCompress Time: {baseTime} + {residualTime} = {baseTime + residualTime}ms\t Decompress Time: {decBasetime} + {self.decompResTime} = {self.decompBaseTime +self.decompResTime}ms")
                # print(f"Epsilon: {epsilonPct }\tCompression Ratio: {CR:.5f}\t Residual CR: {ResidualCR:.5f}\tCompress Time: {baseTime} + {residualTime} = {baseTime + residualTime}ms\t Decompress Time: {decBasetime} + {self.decompResTime} = {self.decompBaseTime +self.decompResTime}ms  \tRange: {ts.range :.3f}")
                # print(f"baseSize: {baseSize/1024 :.3f}KB \t residualSize: {residualSize/1024 :.3f}KB \t origibaseSize: {origibaseSize/1024}KB")
                meanCR += CR
                meanResCR += ResidualCR
                meanTime += residualTime
                meanDec += self.decompResTime
            meanCR, meanTime = meanCR/len(epsilons), meanTime/len(epsilons)
            meanDec = (meanDec+self.decompBaseTime)/len(epsilons)
            meanResCR = meanResCR/len(epsilons)
            meanSpeed = (ts.size/1024/1024)/(meanTime/1000)
            meanDecSpeed = (ts.size/1024/1024)/(meanDec/1000)
            print(f"The average comprssion ratio: {meanCR:.3f},  The average Residual comprssion ratio: {meanResCR:.3f}  average compresstime: {meanTime:.1f}ms,  average Decompresstime: {meanDec:.1f}ms average speed: {meanSpeed:.2f}MB/s, average DeCompress speed: {meanDecSpeed:.2f}MB/s")
            print()


    def main(self) -> None:
        """
        The main function of the whole project
        
        Parameters:
            None
        """


        epsilons = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        filenames = ["FaceFour.csv", "MoteStrain.csv", "Lightning.csv", "Ecg.csv", "Cricket.csv",  
                    "WindDirection.csv", "Wafer.csv", "WindSpeed.csv",  "Pressure.csv"]
        self.run(filenames, epsilons)



if __name__ ==  '__main__':
    test = TestSHRINK()
    test.main()
    print("Congratulation! All test have passed!")