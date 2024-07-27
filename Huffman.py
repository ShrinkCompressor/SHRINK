import heapq
from collections import defaultdict

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    frequency = defaultdict(int)
    for v in data:
        frequency[v] += 1

    priority_queue = [Node(char, freq) for char, freq in frequency.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(priority_queue, merged)

    return priority_queue[0]

def build_huffman_codes(node, current_code, huffman_codes):
    if node:
        if node.char:
            huffman_codes[node.char] = current_code
        build_huffman_codes(node.left, current_code + "0", huffman_codes)
        build_huffman_codes(node.right, current_code + "1", huffman_codes)

def huffman_compress(data):
    root = build_huffman_tree(data)
    huffman_codes = {}
    build_huffman_codes(root, "", huffman_codes)
    print("over......")
    compressed_data = "".join(huffman_codes[char] for char in data)
    return compressed_data, root, huffman_codes

def huffman_decompress(compressed_data, root):
    current_node = root
    decompressed_data = []
    for bit in compressed_data:
        if bit == "0":
            current_node = current_node.left
        else:
            current_node = current_node.right
        if current_node.char:
            decompressed_data . append(current_node.char)
            current_node = root
    return decompressed_data 


def generated_list(low=-20, high=20, size=100):
    import numpy as np
    # Generate a list of floats with a mean of 0 and a uniform distribution between -2 and 2
    generated_list = np.random.randint(low, high, size)
    return generated_list


if __name__ ==  '__main__':
    data = generated_list(size=10)
    data =[10,10,10,10,10,10,10, 1,1,2,1,0]
    data = [str(item) for item in data]
    print("Original data: ", data)
    compressed_data, tree, codes = huffman_compress(data)
    #print(f"Compressed text: {compressed_data}")
    decompressed_data = huffman_decompress(compressed_data, tree)
    decompressed_data = [int(item) if isinstance(item, str) else item for item in decompressed_data ]
    #print(f"Decompressed text: {decompressed_data}")
    print("compression ratio: ", (len(data)*32)/len(compressed_data))
    print(codes)
