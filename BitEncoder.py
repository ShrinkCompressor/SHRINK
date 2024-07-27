import numpy as np


class BitEncoder:
    @staticmethod
    def unpack_data(data, m):
        """
        Unpack an array of data into a binary array. For floating point data, there is also
        an intermediate reinterpretation of the underlying binary data as integer format.

        Args:
            data (array): Array to unpack.
            m (int): Bit depth. For integer data, this can be any positive integer. For
                floating point data, it can be either 32 or 64

        Returns:
            array: Unpacked data
            bool: Indicator for whether original data is floating point
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        data_unpack = np.ascontiguousarray(data, dtype=">i8")  # big endian
        data_unpack = data_unpack.view(">u1")  # split into individual bytes
        data_unpack = np.unpackbits(data_unpack).reshape((*data.shape, -1))  # unpack
        data_unpack = data_unpack[..., -m:].reshape((data.shape[0], -1))  # only m last bits

            # Function to remove leading zeros
        def remove_leading_zeros(arr):
            if np.all(arr == 0):
                return np.array([0], dtype=np.uint8)        
            first_one_index = np.argmax(arr)
            return arr[first_one_index:]

        # Apply the function to each row
        data_unpack = [remove_leading_zeros(row) for row in data_unpack]
        return data_unpack
    
    @staticmethod
    def decode_data(unpacked_data):
        """
        Decode the unpacked data back into the original integer array.

        Args:
            unpacked_data (list): List of unpacked binary arrays.

        Returns:
            list: Original array of integers.
        """
        original_data = []
        for binary_array in unpacked_data:
            # Convert the binary array back to integer
            if len(binary_array) == 1 and binary_array[0] == 0:
                original_data.append(0)
            else:
                binary_str = ''.join(map(str, binary_array))
                original_data.append(int(binary_str, 2))
        return original_data


if __name__ ==  '__main__':
    data_unpack = BitEncoder.unpack_data([0,1,5],3)
    print(data_unpack)
    print(BitEncoder.decode_data(data_unpack))
