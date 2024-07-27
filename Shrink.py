import math
import io
import os
import sys
import csv
import numpy as np
from io import BytesIO
from FloatEncoder import FloatEncoder
from UIntEncoder import UIntEncoder
from VariableByteEncoder import VariableByteEncoder
from Point import Point
from ShrinkSegment import ShrinkSegment
from typing import List
from utilFunction import *
from decimal import Decimal
sys.path.append('/home/guoyou/Lossless')
import QuanTRC
import time


class Shrink:
    def __init__(self, points=None, epsilon=None, bytes=None, variable_byte=False, zstd=False):
        """
        Args:
            points: List[Point]
            epsilon：  ts.range * epsilonPct(0.05)
            bytes：bytes=binary

        Returns:
            
        """
        if points is not None: # Handle the case where points is a list of Points
            if not points:
                raise ValueError("No points provided")
            #self.alpha = 0.05
            #self.alpha = 0.02#cricket
            self.alpha = 0.01
            self.epsilon = epsilon
            self.lastTimeStamp = points[-1].timestamp
            self.values = [point.value for point in points]
            self.max = max(self.values)
            self.min = min(self.values)
            self.length = len(points)
            self.lengthofSegments = None
            self.segments = self.mergePerB(self.compress(points))
            self.points = points[:]
        elif bytes is not None: # Handle the case where bytes is a byte array
            self.readByteArray(bytes, variable_byte, zstd)
        else:
            raise ValueError("Either points or bytes must be provided")

        
    def getResiduals(self):
        self.segments.sort(key=lambda segment: segment.get_init_timestamp)
        residuals = []
        expectedVals = []
        ExpectedPoints = []
        idx = 0
        currentTimeStamp = self.segments[0].get_init_timestamp

        for i in range(len(self.segments) - 1):
            while currentTimeStamp < self.segments[i + 1].get_init_timestamp:
                expectedValue = highPrecisionAdd( self.segments[i].get_a * (currentTimeStamp - self.segments[i].get_init_timestamp), self.segments[i].get_b )
                expectedVals.append(expectedValue)
                residualVal = highPrecisionsubtract(self.values[idx], expectedValue)
                residuals.append(residualVal)
                ExpectedPoints.append(Point(currentTimeStamp, expectedValue))
                currentTimeStamp += 1
                idx += 1

        while currentTimeStamp <= self.lastTimeStamp:
                expectedValue = highPrecisionAdd( self.segments[-1].get_a * (currentTimeStamp - self.segments[-1].get_init_timestamp), self.segments[-1].get_b )
                expectedVals.append(expectedValue)
                residuals.append(highPrecisionsubtract(self.values[idx], expectedValue))
                ExpectedPoints.append(Point(currentTimeStamp, expectedValue))
                currentTimeStamp += 1
                idx += 1    

        csv_file_path = "/home/guoyou/ExtractSemantic/residuals/resdiauls.csv"
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for item in residuals:
                csv_writer.writerow([item])

        return residuals

    
    def residualEncode(self, residuals, epsilon):
        if(epsilon!=0):
            QuantiresdiaulsVals = [round((v/epsilon)) for v in residuals]
        else:
            QuantiresdiaulsVals = residuals[:]

        outputPath = '/home/guoyou/ExtractSemantic/residuals'
        InFilePath = "/home/guoyou/ExtractSemantic/residuals/QuantiresdiaulsVal.csv"
        with open(InFilePath, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for item in QuantiresdiaulsVals:
                csv_writer.writerow([item])

        QuanTRC.compress(InFilePath,outputPath)
        residualSize = os.path.getsize('/home/guoyou/ExtractSemantic/residuals/codes.rc')
        return residualSize
    
    def residualDecode(self, outputPath, epsilon):
        start_time = time.time()
        QuanTRC.decompress(outputPath+ '/codes'+'.rc', outputPath+"/Neworginal"+'.csv')
        Dequant_val = []
        with open( outputPath+"/Neworginal"+'.csv', mode='r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)     
            for row in csv_reader:
                Dequant_val.append(float(row[0]))

        if(epsilon!=0):
            Dequant_val = deResQuantize(Dequant_val, epsilon)
        
        end_time = time.time()
        decompResTime = int((end_time - start_time) * 1000)

        csv_file_path = "/home/guoyou/ExtractSemantic/residuals/DeQuanresdiauls.csv"
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for item in Dequant_val:
                csv_writer.writerow([item])

        return Dequant_val,decompResTime 
        
        
    def AdaptiveMerge(self, points=None, epsilon=None, bytes=None, variable_byte=False, zstd=False):
        """
        Args:
            points: 时序数据的list，内包含point
            epsilon： error threhold, ts.range * epsilonPct
            bytes：

        Returns:
            list，内包含各个compression ratio
        """
        if not points:
            raise ValueError("No points provided")
        
        self.epsilon = epsilon
        self.lastTimeStamp = points[-1].timestamp

        self.segments = None
        Allsegments = []
        CR = []

        currentIdx, preIdx = 0, 0
        partitionPoint = 90000
        
        while(currentIdx < len(points)):
            if(currentIdx-preIdx>=partitionPoint):
                binary = self.toByteArray(variableByte=False, zstd=False)
                compressedSize = len(binary)
                CR.append(((currentIdx-0) * (4 + 4)) / compressedSize)
                Allsegments = self.mergePerB(Allsegments)
                self.segments = Allsegments[:]
                preIdx = currentIdx
            segments = []        
            currentIdx = self.createSegment(currentIdx, points, segments)
            Allsegments.extend(segments)
            self.segments = Allsegments[:]
        
        Allsegments = self.mergePerB(Allsegments)
        self.segments = Allsegments[:]
        binary = self.toByteArray(variableByte=False, zstd=False)
        compressedSize = len(binary)

        print(f"Compression Ratio: {(len(points)  * (4 + 4)) / compressedSize:.3f}")
        CR.append((len(points) * (4 + 4)) / compressedSize)

        return CR
    
    """    
    def ResQuantize(self, x, epsilon):
        return [round((v/epsilon)) for v in x]

    def deResQuantize(self, x_quant, epsilon):
        return [v*epsilon for v in x_quant]
    """
        

    def dynamicEpsilon(self, startIdx, points):
        buflength = int(self.length * self.alpha * self.epsilon) 
        if(buflength >= len(points)):
            return self.epsilon, len(points)-1
        buf = []
        for i in range(startIdx, startIdx+buflength):
            if(i>=len(points)):
                break
            buf.append(points[i].value)

        local_max = max(buf)
        local_min = min(buf)
        C = round(math.exp((2/3 - (local_max - local_min)/ (self.max-self.min))), 1)
        localepsilon = round(self.epsilon * C, 1)

        return localepsilon, startIdx+buflength

    
    def quantization(self, value, localEpsilon):
        res = round(value / localEpsilon ) * localEpsilon
        return res
    
    def createSegment(self, startIdx, points, segments, localEpsilon):
        initTimestamp = points[startIdx].timestamp
        b = self.quantization(points[startIdx].value, localEpsilon)
        

        if startIdx + 1 == len(points): # Case1: only 1 ponit
            segments.append(ShrinkSegment(initTimestamp, -math.inf, math.inf, b))
            return startIdx + 1
        
        aMax = ((points[startIdx + 1].value + localEpsilon) - b) / (points[startIdx + 1].timestamp - initTimestamp)
        aMin = ((points[startIdx + 1].value - localEpsilon) - b) / (points[startIdx + 1].timestamp - initTimestamp)
        if startIdx + 2 == len(points): # Case2: only 2 ponits
            segments.append(ShrinkSegment(initTimestamp, aMin, aMax, b))
            return startIdx + 2
        
        for idx in range(startIdx + 2, len(points)): # Case3: more than 2 ponits
            upValue = points[idx].value + localEpsilon
            downValue = points[idx].value - localEpsilon

            up_lim = aMax * (points[idx].timestamp - initTimestamp) + b
            down_lim = aMin * (points[idx].timestamp - initTimestamp) + b

            if (downValue > up_lim or upValue < down_lim):
                segments.append(ShrinkSegment(initTimestamp, aMin , aMax, b))
                return idx
            
            if upValue < up_lim:
                aMax = max((upValue - b) / (points[idx].timestamp - initTimestamp), aMin)
            if downValue > down_lim:
                aMin = min((downValue - b) / (points[idx].timestamp - initTimestamp), aMax)
            

        segments.append(ShrinkSegment(initTimestamp, aMin, aMax, b))

        return len(points)
            
    def compress(self, points):
        segments = []
        currentIdx = 0
        newIdx = -1
        localEpsilon = self.epsilon
        
        while(currentIdx < len(points)):
            if(currentIdx>newIdx):
                localEpsilon, newIdx  = self.dynamicEpsilon(currentIdx, points)
            currentIdx = self.createSegment(currentIdx, points, segments, localEpsilon)

        return segments

        
    def mergePerB(self, segments):
        aMinTemp = float('-inf')
        aMaxTemp = float('inf')
        b = float('nan')
        timestamps = []
        mergedSegments = []
        self.lengthofSegments = 0

        segments.sort(key=lambda segment: (segment.get_b, segment.get_a))
        
        for i in range(len(segments)):
            if b != segments[i].get_b:
                if len(timestamps) == 1:
                    mergedSegments.append(ShrinkSegment(timestamps[0], aMinTemp, aMaxTemp, b))
                    self.lengthofSegments += 1
                else:
                    for timestamp in timestamps:
                        mergedSegments.append(ShrinkSegment(timestamp, aMinTemp, aMaxTemp, b))
                        self.lengthofSegments += 1

                
                timestamps.clear()
                timestamps.append(segments[i].get_init_timestamp)
                aMinTemp = segments[i].get_a_min
                aMaxTemp = segments[i].get_a_max
                b = segments[i].get_b
                continue
            
            if segments[i].get_a_min <= aMaxTemp and segments[i].get_a_max >= aMinTemp:
                timestamps.append(segments[i].get_init_timestamp)
                aMinTemp = max(aMinTemp, segments[i].get_a_min)
                aMaxTemp = min(aMaxTemp, segments[i].get_a_max)
            else:
                if len(timestamps) == 1:
                    mergedSegments.append(segments[i - 1])
                    self.lengthofSegments += 1
                else:
                    for timestamp in timestamps:
                        mergedSegments.append(ShrinkSegment(timestamp, aMinTemp, aMaxTemp, b))
                        self.lengthofSegments += 1
                
                timestamps.clear()
                timestamps.append(segments[i].get_init_timestamp)
                aMinTemp = segments[i].get_a_min
                aMaxTemp = segments[i].get_a_max
        
        if timestamps:
            if len(timestamps) == 1:
                mergedSegments.append(ShrinkSegment(timestamps[0], aMinTemp, aMaxTemp, b))
                self.lengthofSegments += 1

            else:
                for timestamp in timestamps:
                    mergedSegments.append(ShrinkSegment(timestamp, aMinTemp, aMaxTemp, b))
                    self.lengthofSegments += 1
        
        return mergedSegments
    

    def decompress(self):
        start_time = time.time()
        
        # Pre-calculate the initial timestamps and other values
        init_timestamps = [segment.get_init_timestamp for segment in self.segments]
        a_values = [segment.a for segment in self.segments]
        b_values = [segment.get_b for segment in self.segments]
        points = []

        # Loop over segments, avoiding method calls within the loop
        for i in range(len(self.segments) - 1):
            timestamps = range(init_timestamps[i], init_timestamps[i + 1])
            points += [Point(ts, a_values[i] * (ts - init_timestamps[i]) + b_values[i]) for ts in timestamps]

        # Handle the last segment
        last_segment_timestamps = range(init_timestamps[-1], self.lastTimeStamp + 1)
        points += [Point(ts, a_values[-1] * (ts - init_timestamps[-1]) + b_values[-1]) for ts in last_segment_timestamps]

        end_time = time.time()
        decompBaseTime = int((end_time - start_time) * 1000)
        return points, decompBaseTime
    



        
    def toByteArrayPerBSegments(self, segments: List[ShrinkSegment], variableByte: bool, outStream: io.BytesIO) -> None:
        # Initialize a dictionary to organize segments by 'b' value
        input = {}
        resArr = []

        for segment in segments:
            a = segment.get_a
            b = round(segment.get_b / self.epsilon)
            t = segment.get_init_timestamp
            
            if b not in input:
                input[b] = {}
            
            if a not in input[b]:
                input[b][a] = []
            
            input[b][a].append(t)
        
        # Write the size of the dictionary
        VariableByteEncoder.write(len(input), outStream)
        resArr.append(len(input))###****需要删除****###
        
        if not input.items():
            return
        
        previousB = min(input.keys())
        VariableByteEncoder.write(previousB, outStream)
        resArr.append(previousB)###****需要删除****###
        
        for b, aSegments in input.items():
            VariableByteEncoder.write(b - previousB, outStream)
            resArr.append(b - previousB)###****需要删除****###
            previousB = b
            VariableByteEncoder.write(len(aSegments), outStream)
            resArr.append(len(aSegments))###****需要删除****###

            
            for a, timestamps in aSegments.items():
                # Custom method to encode the float 'a' value
                FloatEncoder.write(float(a), outStream)
                resArr.append(float(a))###****需要删除****###
                len(aSegments)
                
                if variableByte:
                    print("variableByte为True了，出现错误\n")
                    timestamps.sort()
                
                VariableByteEncoder.write(len(timestamps), outStream)
                resArr.append(len(timestamps))###****需要删除****###

                
                previousTS = 0
                
                for timestamp in timestamps:
                    if variableByte:
                        print("variableByte为True了，出现错误\n")
                        VariableByteEncoder.write(timestamp - previousTS, outStream)
                        resArr.append(timestamp - previousTS)###****需要删除****###
                    else:
                        # Custom method to write 'timestamp' as an unsigned int
                        UIntEncoder.write(timestamp, outStream)
                        resArr.append(timestamp)###****需要删除****###

                        
                    
                    previousTS = timestamp
        np_array = np.array(resArr, dtype=np.float32)
        np.save('/home/guoyou/ExtractSemantic/Base/'+" Base_Watch_accelerometer.npy", np_array)

        
    def toByteArray(self, variableByte: bool, zstd: bool) -> bytes:
        outStream = BytesIO()
        bytes = None

        FloatEncoder.write(float(self.epsilon), outStream)

        self.toByteArrayPerBSegments(self.segments, variableByte, outStream)

        if variableByte:
            VariableByteEncoder.write(int(self.lastTimeStamp), outStream)
        else:
            UIntEncoder.write(self.lastTimeStamp, outStream)

        if zstd:
            bytes = zstd.compress(outStream.getvalue())
        else:
            bytes = outStream.getvalue()

        outStream.close()
        return bytes
    
    def saveByte(self, byts, filename):
        
        path = '/home/guoyou/ExtractSemantic/Base/'+filename[:-7]+"_Base.bin"
        with open(path, 'wb') as file:
            file.write(byts)

        baseSize = os.path.getsize(path)
        return baseSize

        
    def readMergedPerBSegments(self, variableByte, inStream):
        segments = []
        numB = VariableByteEncoder.read(inStream)

        if numB == 0:
            return segments

        previousB = VariableByteEncoder.read(inStream)

        for _ in range(numB):
            b = VariableByteEncoder.read(inStream) + previousB
            previousB = b
            numA = VariableByteEncoder.read(inStream)

            for _ in range(numA):
                a = FloatEncoder.read(inStream)
                numTimestamps = VariableByteEncoder.read(inStream)

                for _ in range(numTimestamps):
                    if variableByte:
                        timestamp += VariableByteEncoder.read(inStream)
                    else:
                        timestamp = UIntEncoder.read(inStream)
                    segments.append(ShrinkSegment(timestamp, a, a, b * self.epsilon))

        return segments

    def readByteArray(self, input, variableByte, zstd):
        if zstd:
            binary = zstd.decompress(input)
        else:
            binary = input

        inStream = BytesIO(binary)

        self.epsilon = FloatEncoder.read(inStream)
        self.segments = self.readMergedPerBSegments(variableByte, inStream)

        if variableByte:
            self.lastTimeStamp = VariableByteEncoder.read(inStream)
        else:
            self.lastTimeStamp = UIntEncoder.read(inStream)

        inStream.close()
