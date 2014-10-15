#from numpy import *
import pycuda.driver as cuda
import matrixio
import Image
import numpy

def create_2d_texture(a, module, variable, point_sampling=False):
    a = numpy.ascontiguousarray(a)
    out_texref = module.get_texref(variable)
    cuda.matrix_to_texref(a, out_texref, order='C')
    if point_sampling: out_texref.set_filter_mode(cuda.filter_mode.POINT)
    else: out_texref.set_filter_mode(cuda.filter_mode.LINEAR)
    return out_texref

        
def create_2d_rgba_texture(a, module, variable, point_sampling=False):
    a = numpy.ascontiguousarray(a)
    out_texref = module.get_texref(variable)
    cuda.bind_array_to_texref(
        cuda.make_multichannel_2d_array(a, order='C'), out_texref)    
    if point_sampling: out_texref.set_filter_mode(cuda.filter_mode.POINT)
    else: out_texref.set_filter_mode(cuda.filter_mode.LINEAR)
    return out_texref


def create_3d_texture(a, module, variable, point_sampling=False):
    
    a = numpy.asfortranarray(a)
    w, h, d = a.shape
    
    descr = cuda.ArrayDescriptor3D()
    descr.width = w
    descr.height = h
    descr.depth = d
    descr.format = cuda.dtype_to_array_format(a.dtype)
    descr.num_channels = 1
    descr.flags = 0
    ary = cuda.Array(descr)
   
    copy = cuda.Memcpy3D()
    copy.set_src_host(a)
    copy.set_dst_array(ary)
    copy.width_in_bytes = copy.src_pitch = a.strides[1]
    copy.src_height = copy.height = h
    copy.depth = d
    copy()

    out_texref = module.get_texref(variable)
    out_texref.set_array(ary)
    if point_sampling: 
        out_texref.set_filter_mode(cuda.filter_mode.POINT)
    else: 
        out_texref.set_filter_mode(cuda.filter_mode.LINEAR)
    return out_texref
    

def create_4d_texture(a, module, variable, point_sampling=False):
    """Bind a 3D CUDA texture from a 4D numpy array, return pair
       (texture reference, original shape). The returned reference is 3D,
       since there are no 4D textures in CUDA."""
    a = numpy.asfortranarray(a)
    w, h, d, f = a.shape
    return create_3d_texture(reshape(a, (w, h, d*f), order='F'),
        module, variable, point_sampling), a.shape


def update_2d_texture(texref, newdata):
    arr = texref.get_array()
    newdata = numpy.ascontiguousarray(newdata)
    h, w = newdata.shape
    
    desc = arr.get_descriptor()
    assert h == desc.height and w == desc.width
    assert desc.num_channels == 1
    
    copy = cuda.Memcpy2D()
    copy.set_src_host(newdata)
    copy.set_dst_array(arr)
    copy.width_in_bytes = copy.src_pitch = newdata.strides[0]
    copy.src_height = copy.height = h
    copy(True)
    
    

# def load_data_1d(filename, name, mod):
#     data = read_1d_data(filename)
#     texref = create_1d_texture(data, mod, name)
#     return texref, data.shape


# def load_data_2d(filename, name, mod):
#     data = read_2d_data(filename)
#     texref = create_2d_texture(data, mod, name)
#     return texref, data.shape

def load_data_2d(filename, name, mod):
    data = matrixio.read_2d_data(filename)
   # print 'load_data_2d:', filename, 'dtype:', data.dtype, 'shape:', data.shape
    assert len(data.shape) == 2
    texref = create_2d_texture(data, mod, name)
    return texref, data.shape
    
    
def load_data_2d_rgba(filename, name, mod):
    # data = asarray(Image.open(filename))
    data = matrixio.read_2d_data(filename)
#    print 'load_data_2d_rgba:', filename, 'dtype:', data.dtype, 'shape:', data.shape

    assert len(data.shape) == 3
    assert data.shape[2] == 3 or data.shape[2] == 4
    
    if data.shape[2] == 3:
        data = dstack([data, zeros(data.shape[0:2], dtype=a.dtype)])

    texref = create_2d_rgba_texture(data, mod, name)
    return texref, data.shape


def load_data_3d(filename, name, mod, rank, data_divide):
    data = matrixio.read_3d_data(filename)
    texref = create_3d_texture(data, mod, name)
    return texref, data.shape


def load_data_4d(filename, name, mod):
    data = matrixio.read_4d_data(filename)
    print 'load_data_4d:', filename, 'dtype:', data.dtype, 'shape:', data.shape
    texref, shape = create_4d_texture(data, mod, name)
    return texref, shape
