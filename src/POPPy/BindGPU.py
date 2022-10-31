import ctypes
import math
import numpy as np
import sys

from src.POPPy.BindUtils import *
from src.POPPy.Structs import *
from src.POPPy.POPPyTypes import *

#############################################################################
#                                                                           #
#              List of bindings for the GPU interface of POPPy.             #
#                                                                           #
#############################################################################

def loadGPUlib():
    lib = ctypes.CDLL('./src/C++/libpoppygpu.so')

    lib.callKernelf_JM.argtypes = [ctypes.POINTER(c2Bundlef), reflparamsf, reflparamsf,
                                   ctypes.POINTER(reflcontainerf), ctypes.POINTER(reflcontainerf),
                                   ctypes.POINTER(c2Bundlef), ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callKernelf_JM.restype = None

    lib.callKernelf_EH.argtypes = [ctypes.POINTER(c2Bundlef), reflparamsf, reflparamsf,
                                   ctypes.POINTER(reflcontainerf), ctypes.POINTER(reflcontainerf),
                                   ctypes.POINTER(c2Bundlef), ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callKernelf_EH.restype = None

    lib.callKernelf_JMEH.argtypes = [ctypes.POINTER(c4Bundlef), reflparamsf, reflparamsf,
                                   ctypes.POINTER(reflcontainerf), ctypes.POINTER(reflcontainerf),
                                   ctypes.POINTER(c2Bundlef), ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callKernelf_JMEH.restype = None

    lib.callKernelf_EHP.argtypes = [ctypes.POINTER(c2rBundlef), reflparamsf, reflparamsf,
                                   ctypes.POINTER(reflcontainerf), ctypes.POINTER(reflcontainerf),
                                   ctypes.POINTER(c2Bundlef), ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callKernelf_EHP.restype = None

    lib.callKernelf_FF.argtypes = [ctypes.POINTER(c2Bundlef), reflparamsf, reflparamsf,
                                   ctypes.POINTER(reflcontainerf), ctypes.POINTER(reflcontainerf),
                                   ctypes.POINTER(c2Bundlef), ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callKernelf_FF.restype = None
    return lib

# WRAPPER FUNCTIONS DOUBLE PREC

#### SINGLE PRECISION
def POPPy_GPUf(source, target, currents, k, epsilon, t_direction, nThreads, mode):
    lib = loadGPUlib()

    # Create structs with pointers for source and target
    csp = reflparamsf()
    ctp = reflparamsf()

    cs = reflcontainerf()
    ct = reflcontainerf()

    gs = source["gridsize"][0] * source["gridsize"][1]
    gt = target["gridsize"][0] * target["gridsize"][1]

    allfill_reflparams(csp, source, ctypes.c_float)
    allfill_reflparams(ctp, target, ctypes.c_float)

    allocate_reflcontainer(cs, gs, ctypes.c_float)
    allocate_reflcontainer(ct, gt, ctypes.c_float)

    if mode == "Scalar":
        c_cfield = arrC1f()
        fieldConv(field, c_field, gs, ctypes.c_float)

    else:
        c_currents = c2Bundlef()
        currentConv(currents, c_currents, gs, ctypes.c_float)

    target_shape = (target["gridsize"][0], target["gridsize"][1])

    nBlocks = math.ceil(gt / nThreads)

    k           = ctypes.c_float(k)
    nThreads    = ctypes.c_int(nThreads)
    nBlocks     = ctypes.c_int(nBlocks)
    epsilon     = ctypes.c_float(epsilon)
    t_direction = ctypes.c_float(t_direction)

    if mode == "JM":
        res = c2Bundlef()
        allocate_c2Bundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        lib.callKernelf_JM(ctypes.byref(res), csp, ctp,
                                ctypes.byref(cs), ctypes.byref(ct),
                                ctypes.byref(c_currents), k, epsilon,
                                t_direction, nBlocks, nThreads)

        # Unpack filled struct
        JM = c2BundleToObj(res, shape=target_shape, obj_t='currents')

        return JM

    elif mode == "EH":
        res = c2Bundlef()
        allocate_c2Bundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        lib.callKernelf_EH(ctypes.byref(res), csp, ctp,
                                ctypes.byref(cs), ctypes.byref(ct),
                                ctypes.byref(c_currents), k, epsilon,
                                t_direction, nBlocks, nThreads)

        # Unpack filled struct
        EH = c2BundleToObj(res, shape=target_shape, obj_t='fields')

        return EH

    elif mode == "JMEH":
        res = c4Bundlef()
        allocate_c4Bundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        lib.callKernelf_JMEH(ctypes.byref(res), csp, ctp,
                                ctypes.byref(cs), ctypes.byref(ct),
                                ctypes.byref(c_currents), k, epsilon,
                                t_direction, nBlocks, nThreads)

        # Unpack filled struct
        JM, EH = c4BundleToObj(res, shape=target_shape)

        return [JM, EH]

    elif mode == "EHP":
        res = c2rBundlef()
        allocate_c2rBundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        lib.callKernelf_EHP(ctypes.byref(res), csp, ctp,
                                ctypes.byref(cs), ctypes.byref(ct),
                                ctypes.byref(c_currents), k, epsilon,
                                t_direction, nBlocks, nThreads)

        # Unpack filled struct
        EH, Pr = c2rBundleToObj(res, shape=target_shape)

        return [EH, Pr]

    elif mode == "FF":
        res = c2Bundlef()
        allocate_c2Bundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        lib.callKernelf_FF(ctypes.byref(res), csp, ctp,
                                ctypes.byref(cs), ctypes.byref(ct),
                                ctypes.byref(c_currents), k, epsilon,
                                t_direction, nBlocks, nThreads)

        # Unpack filled struct
        EH = c2BundleToObj(res, shape=target_shape, obj_t='fields')

        return EH

if __name__ == "__main__":
    print("Bindings for POPPy GPU.")
