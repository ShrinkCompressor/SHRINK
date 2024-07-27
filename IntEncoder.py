import io
import struct
from io import BytesIO


class IntEncoder:
    @staticmethod
    def write(number, output_stream):
        int_bytes = struct.pack('>i', number)
        output_stream.write(int_bytes)
        return int_bytes

    @staticmethod
    def read(input_stream):
        int_bytes = input_stream.read(4)
        number = struct.unpack('>i', int_bytes)[0]
        return number
    
if __name__ ==  '__main__':
    outStream = BytesIO()
    print(IntEncoder.write(0,outStream ))#10,4bytes
    print(len(IntEncoder.write(0,outStream )))#10,4bytes

