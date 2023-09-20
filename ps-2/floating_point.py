import numpy as np

def get_bits(number):
    """
    From course material

    For a NumPy quantity, return bit representation
    
    Inputs:
    ------
    number : NumPy value
        value to convert into list of bits
        
    Returns:
    -------
    bits : list
       list of 0 and 1 values, highest to lowest significance
    """
    bytes = number.tobytes()
    bits = []
    for byte in bytes:
        bits = bits + np.flip(np.unpackbits(np.uint8(byte)), np.uint8(0)).tolist()
    return list(reversed(bits))

def get_val(_bits):
    '''
    get value from bit representation of float32
    '''
    assert(len(_bits) == 32)
    sign_bit = _bits[0] # 1 bit
    exp_bits = _bits[1:9] # 8 bits
    mantissa_bits = _bits[9:] # 23 bits
    mantissa_val = 1
    for i in range(23):
        mantissa_val += mantissa_bits[i] * pow(2, -(1+i))
    exp_val = 0
    for i in range(8):
        exp_val += exp_bits[i] * pow(2, 7 - i)
    return np.longdouble(mantissa_val) * pow(2, exp_val - 127)



f = np.float32(100.98763)
# int32bits = f.view(np.int32)
# exp_bits = int32bits
# print('{:032b}'.format(int32bits)) # print the 32 bit integer in its binary representation
bits_list = get_bits(np.float32(100.98763))
f_new = get_val(bits_list)
print("{value} -> {bitlist}".format(value = 100.98763, bitlist = str(bits_list)))
print(f"new value: {f_new}")
print("difference: {v}".format(v= np.longdouble(f_new) - 100.98763))

