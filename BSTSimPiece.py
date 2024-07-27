import math
import io
from io import BytesIO
from FloatEncoder import FloatEncoder
from UIntEncoder import UIntEncoder
from VariableByteEncoder import VariableByteEncoder
from Point import Point
from SHRINK.ShrinkSegment import SimPieceSegment
from typing import List
from BST import BST
from BSTSimPieceSegment import BSTSimPieceSegment
import itertools




class BSTSimPiece:
    def __init__(self, points=None, epsilon=None, bytes=None, variable_byte=False, zstd=False,residual=None):
        if points is not None:
            # Handle the case where points is a list of Points
            if not points:
                raise ValueError("No points provided")
            self.points = points
            self.epsilon = epsilon
            self.lastTimeStamp = points[-1].timestamp
            #self.segments = self.compress(points)
            self.segments, self.residuals = self.mergePerB(self.compress(points))
        elif bytes is not None:
            # Handle the case where bytes is a byte array
            self.readByteArray(bytes, variable_byte, zstd)
        else:
            raise ValueError("Either points or bytes must be provided")

    
    def quantization(self, value):
        return round(value / self.epsilon) * self.epsilon
    
    def createSegment(self, startIdx, points, segments):
        """
        计算从time startIdx作为起点的segment

        Args:
            startIdx: 初始timestamp
            points: 所有节点list
            segments： dictionary {b: BSTSimPieceSegment(initTimestamp, aMin, aMax)}

        Returns:
            当前segment结束点的timestamp
        """

        initTimestamp = points[startIdx].timestamp
        b = self.quantization(points[startIdx].value)
        
        if startIdx + 1 == len(points): # Case1: only 1 ponit
            if(b not in segments):
                bst = BST()
                bst.insert(BSTSimPieceSegment(initTimestamp, -math.inf, math.inf, b))
                segments[b] = bst
            else:
                segments[b].insert(BSTSimPieceSegment(initTimestamp, -math.inf, math.inf, b))
            return startIdx + 1
        
        aMax = ((points[startIdx + 1].value + self.epsilon) - b) / (points[startIdx + 1].timestamp - initTimestamp)
        aMin = ((points[startIdx + 1].value - self.epsilon) - b) / (points[startIdx + 1].timestamp - initTimestamp)
        if startIdx + 2 == len(points): # Case2: only 2 ponits
            if(b not in segments):
                bst = BST()
                bst.insert(BSTSimPieceSegment(initTimestamp, aMin, aMax, b))
                segments[b] = bst
            else:
                segments[b].insert(BSTSimPieceSegment(initTimestamp, aMin, aMax, b))
            return startIdx + 2
        
        for idx in range(startIdx + 2, len(points)): # Case3: more than 2 ponits
            upValue = points[idx].value + self.epsilon
            downValue = points[idx].value - self.epsilon

            up_lim = aMax * (points[idx].timestamp - initTimestamp) + b
            down_lim = aMin * (points[idx].timestamp - initTimestamp) + b

            if (downValue > up_lim or upValue < down_lim):
                if(b not in segments):
                    bst = BST()
                    bst.insert(BSTSimPieceSegment(initTimestamp, aMin, aMax, b))
                    segments[b] = bst
                else:
                    segments[b].insert(BSTSimPieceSegment(initTimestamp, aMin, aMax, b))
                return idx
            
            if upValue < up_lim:
                aMax = max((upValue - b) / (points[idx].timestamp - initTimestamp), aMin)
            if downValue > down_lim:
                aMin = min((downValue - b) / (points[idx].timestamp - initTimestamp), aMax)
            
        if(b not in segments):
            bst = BST()
            bst.insert(BSTSimPieceSegment(initTimestamp, aMin, aMax, b))
            segments[b] = bst        
        else:
            segments[b].insert(BSTSimPieceSegment(initTimestamp, aMin, aMax, b))     

        return len(points)
            
    def compress(self, points): 
        """
        compress points

        Args:
            points:

        Returns:
            segments = {
                        b1:[BSTSimPieceSegment(initTimestamp1, aMin1, aMax1),BSTSimPieceSegment(initTimestamp2, aMin2, aMax2)],
                        b2:[BSTSimPieceSegment(initTimestamp3, aMin3, aMax3),BSTSimPieceSegment(initTimestamp4, aMin4, aMax4)]
                        }
        """
        segments = {}
        currentIdx = 0
        
        while(currentIdx < len(points)):
            currentIdx = self.createSegment(currentIdx, points, segments)
        
        for b, s in segments.items():
            segments[b] = s.to_list_non_recursive()
            
        return segments
    
    def mergePerB(self, segments):
        aMinTemp = float('-inf')
        aMaxTemp = float('inf')
        b = float('nan')
        timestamps = []
        mergedSegments = []

        ### 增加一步以符合原算法的输入，生成形如mergedSegments = [BSTSimPieceSegment(initTimestamp, aMin, aMax, b)]
        for b, segList in segments.items():
            for seg in segList:
                mergedSegments.append(BSTSimPieceSegment(seg.get_init_timestamp, seg.get_a_min, seg.get_a_max, b))
        segments = mergedSegments[:]
        mergedSegments = []
        ### 增加一步以符合原算法的输入，生成形如mergedSegments = [BSTSimPieceSegment(initTimestamp, aMin, aMax, b)]

        #segments.sort(key=lambda segment: (segment.get_b, segment.get_a))
        segments.sort(key=lambda segment: (segment.get_b))
        
        for i in range(len(segments)):
            if b != segments[i].get_b:
                if len(timestamps) == 1:
                    mergedSegments.append(SimPieceSegment(timestamps[0], aMinTemp, aMaxTemp, b))
                else:
                    for timestamp in timestamps:
                        mergedSegments.append(SimPieceSegment(timestamp, aMinTemp, aMaxTemp, b))
                
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
                else:
                    for timestamp in timestamps:
                        mergedSegments.append(SimPieceSegment(timestamp, aMinTemp, aMaxTemp, b))
                
                timestamps.clear()
                timestamps.append(segments[i].get_init_timestamp)
                aMinTemp = segments[i].get_a_min
                aMaxTemp = segments[i].get_a_max
        
        if timestamps:
            if len(timestamps) == 1:
                mergedSegments.append(SimPieceSegment(timestamps[0], aMinTemp, aMaxTemp, b))
            else:
                for timestamp in timestamps:
                    mergedSegments.append(SimPieceSegment(timestamp, aMinTemp, aMaxTemp, b))

        #**********额外增加的计算residual**********
        mergedSegments.sort(key=lambda segment: segment.get_init_timestamp)
        residuals = []
        currentTimeStamp = mergedSegments[0].get_init_timestamp

        for i in range(len(mergedSegments) - 1):
            while currentTimeStamp < mergedSegments[i + 1].get_init_timestamp:
                value = mergedSegments[i].get_a * (currentTimeStamp - mergedSegments[i].get_init_timestamp) + mergedSegments[i].get_b
                residuals.append(self.points[currentTimeStamp].value- value)
                currentTimeStamp += 1

        while currentTimeStamp <= self.lastTimeStamp:
            value = mergedSegments[-1].get_a * (currentTimeStamp - mergedSegments[-1].get_init_timestamp) + mergedSegments[-1].get_b
            residuals.append(self.points[currentTimeStamp].value- value)
            currentTimeStamp += 1
        #**********额外增加的计算residual**********

        
        return mergedSegments, residuals
    
    def BSTmerge(self, segments):
        for b, root in segments.items():
            segments[b] = root.mergeAll()

        return segments


        
    def decompress(self, residuals):
        self.segments.sort(key=lambda segment: segment.get_init_timestamp)
        points = []
        currentTimeStamp = self.segments[0].get_init_timestamp

        for i in range(len(self.segments) - 1):
            while currentTimeStamp < self.segments[i + 1].get_init_timestamp:
                value = self.segments[i].get_a * (currentTimeStamp - self.segments[i].get_init_timestamp) + self.segments[i].get_b
                points.append(Point(currentTimeStamp, value+residuals[currentTimeStamp]))
                currentTimeStamp += 1

        while currentTimeStamp <= self.lastTimeStamp:
            value = self.segments[-1].get_a * (currentTimeStamp - self.segments[-1].get_init_timestamp) + self.segments[-1].get_b
            points.append(Point(currentTimeStamp, value+residuals[currentTimeStamp]))
            currentTimeStamp += 1

        return points
    
    def toByteArrayPerBSegments(self, segments: List[BSTSimPieceSegment], variableByte: bool, outStream: io.BytesIO) -> None:
        # Initialize a dictionary to organize segments by 'b' value
        input = {}
        
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
        
        if not input.items():
            return
        
        previousB = min(input.keys())
        VariableByteEncoder.write(previousB, outStream)
        
        for b, aSegments in input.items():
            VariableByteEncoder.write(b - previousB, outStream)
            previousB = b
            VariableByteEncoder.write(len(aSegments), outStream)
            
            for a, timestamps in aSegments.items():
                # Custom method to encode the float 'a' value
                FloatEncoder.write(float(a), outStream)
                
                if variableByte:
                    print("variableByte为True了，出现错误\n")
                    timestamps.sort()
                
                VariableByteEncoder.write(len(timestamps), outStream)
                
                previousTS = 0
                
                for timestamp in timestamps:
                    if variableByte:
                        print("variableByte为True了，出现错误\n")
                        VariableByteEncoder.write(timestamp - previousTS, outStream)
                    else:
                        # Custom method to write 'timestamp' as an unsigned int
                        UIntEncoder.write(timestamp, outStream)
                    
                    previousTS = timestamp

        
    def toByteArray(self, variableByte: bool, zstd: bool) -> bytes:
        outStream = BytesIO()
        bytes = None

        FloatEncoder.write(float(self.epsilon), outStream)

        self.toByteArrayPerBSegments(self.segments, variableByte, outStream)

        if variableByte:
            print("variableByte为True了，出现错误\n")
            VariableByteEncoder.write(int(self.lastTimeStamp), outStream)
        else:
            UIntEncoder.write(self.lastTimeStamp, outStream)

        if zstd:
            print("zstd为True了，出现错误\n")
            bytes = zstd.compress(outStream.getvalue())
        else:
            bytes = outStream.getvalue()

        outStream.close()
        return bytes

        
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
                    segments.append(SimPieceSegment(timestamp, a, a, b * self.epsilon))

        return segments

    def readByteArray(self, input, variableByte, zstd):
        if zstd:
            binary = zstd.decompress(input)
        else:
            binary = input

        inStream = BytesIO(binary)

        self.epsilon = FloatEncoder.read(inStream)
        #print("epsilon  = ", self.epsilon)
        self.segments = self.readMergedPerBSegments(variableByte, inStream)

        if variableByte:
            print("variableByte为True了，出现错我\n")
            self.lastTimeStamp = VariableByteEncoder.read(inStream)
        else:
            self.lastTimeStamp = UIntEncoder.read(inStream)

        inStream.close()
