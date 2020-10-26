
# Imports
import math
import os

# Static variables
ALPHABET = ''

# Function declaration

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

#test for exact equality
def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

def encode(n):
    try:
        return ALPHABET[n]
    except IndexError:
        raise Exception ("cannot encode: %s" % n)

def dec_to_base(dec = 0, base = 16):
    if dec < base:
        return encode (dec)
    else:
        return dec_to_base (dec // base, base) + encode (dec % base)