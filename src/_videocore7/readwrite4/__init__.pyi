from ctypes import c_uint32, c_void_p

def read4(addr: c_void_p) -> c_uint32: ...
def write4(addr: c_void_p, data: c_uint32) -> None: ...
