#from numpy import *
from scipy.io import loadmat, savemat
#import os, dicom, Image, gzip
import os, Image, gzip
import numpy

def read_mtx(filename):
    f = open(filename, "rb")
    
    # read size
    m, n = fread(f, 2, 'i')
    
    # read /ata
    A = fread(f, m*n, 'f')
    A = numpy.reshape(A, (m, n))
    
    f.close()
    return A

    
def write_mtx(A, filename):
    assert A.dtype == "float32"
    A = ascontiguousarray(A)
    f = open(filename, "wb")
    
    # write size
    m, n = A.shape
    size = array([m, n], dtype=int32)
    f.write(str(size.data))
    
    # write data
    f.write(str(A.data))

    f.close()


def read_matlab(filename):
    d = loadmat(filename)
    xs = [x for (x, y) in d.items() if not x.startswith('__')]
    assert len(xs) == 1, "only one variable can be defined in " + filename
    return d[xs[0]]
    

def write_matlab(A, filename):
    savemat(filename, {"data": A})


def read_numpy(filename):
    if filename.endswith('.npz'):
        npz = load(filename)
        # fixed by cpbotha (info@charlboth.com) for numpy 1.3, which does 
        # NOT have keys/items ivars
        # 1. npz.keys() -> npz.files
        assert len(npz.files) == 1, "read_numpy: only one variable should be defined in " + filename
        # 2. npz.items()[0][1] -> npz[npz.files[0]]
      #  print "npz file shape: ",npz[npz.files[0]].shape
        return npz[npz.files[0]]
    elif filename.endswith('.npy'):
        return load(filename)
    else:
        assert False, "read_numpy: wrong file extension: " + filename


def read_nrrd(filename):
    f = open(filename)
    
    # read header info
    magic = f.readline()
    assert magic.startswith('NRRD')
    info = {}
    
    line = f.readline().strip()
    while len(line) > 0:
        if not line.startswith('#'):
            print line
            print len(line)
            pos = line.find(':')
            assert pos >= 0
            info[line[:pos]] = line[pos+1:].strip()
        line = f.readline().strip()
    
    ndim = int(info['dimension'])
    shape = tuple(map(int, info['sizes'].split()))
    assert len(shape) == ndim

    # get data type
    dt = info['type']
    if dt in ["signed char", "int8", "int8_t"]:
        dt = 'int8'
    elif dt in ["uchar", "unsigned char", "uint8", "uint8_t"]:
        dt = 'uint8'
    elif dt in ["short", "short int", "signed short", "signed short int", "int16", "int16_t"]:
        dt = 'int16'
    elif dt in ["ushort", "unsigned short", "unsigned short int", "uint16", "uint16_t"]:
        dt = 'uint16'
    elif dt in ["ushort", "unsigned short", "unsigned short int", "uint16", "uint16_t"]:
        dt = 'uint16'
    elif dt in ["int", "signed int", "int32", "int32_t"]:
        dt = 'int32'
    elif dt in ["uint", "unsigned int", "uint32", "uint32_t"]:
        dt = 'uint32'
    elif dt in ["longlong", "long long", "long long int", "signed long long", "signed long long int", "int64", "int64_t"]:
        dt = 'int64'
    elif dt in ["ulonglong", "unsigned long long", "unsigned long long int", "uint64", "uint64_t"]:
        dt = 'uint64'
    elif dt == 'float':
        dt = 'float32'
    elif dt == 'double':
        dt = 'float64'
    else:
        assert False, "read_nrrd: unknown type: " + dt
        
    dt = dtype(dt)
    
    # read data
    if info.has_key('endian'):
        assert info['endian'] == 'little'
    
    if info['encoding'] == 'raw':
        A = numpy.fromfile(f, dt, product(shape))
    elif info['encoding'] in ['txt', 'text', 'ascii']:
        A = numpy.fromfile(f, dt, product(shape), ' ')
    elif info['encoding'] in ['gz', 'gzip']:
        g = gzip.GzipFile(None, 'rb', 9, f)
        s = g.read(product(shape) * dt.itemsize)
        A = fromstring(s, dt)
        g.close()
    else:
        assert False, "read_nrrd: unknown encoding"
    
    f.close()
    A = nupy.reshape(A, shape, order='F')
    return A 

def read_dat(filename, out_of_core= False):
    #read dat
    #print filename, filename.rfind('/'), filename[0:filename.rfind('/')]

	print "A"
	print filename

	f = open(filename)
	file_info = {}
	line = f.readline().strip()
	while len(line) > 0:
		if not line.startswith('#'):
			pos = line.find(':')
			assert(pos >= 0)
			key = line[:pos].strip()
			value = line[pos+1:].strip()		
			file_info[key] = value

		line = f.readline().strip()

	if out_of_core:
		return file_info

	ndim = int(3)
	raw_filename = file_info['FileName']
	shape = file_info['Resolution'].split()
	if len(shape) == 3: shape[0],shape[1],shape[2] = int(shape[2]),int(shape[1]),int(shape[0])
	if len(shape) == 2: shape[0],shape[1] = int(shape[1]),int(shape[0])
	format = file_info['Format'].lower()

	#read raw
	print "read dat file info"
	if format == 'uchar': format = numpy.uint8
	print filename[0:filename.rfind('/')+1]+raw_filename, format, shape 
	data = numpy.fromfile(filename[0:filename.rfind('/')+1]+raw_filename,dtype=format)
	data = numpy.reshape(data,shape)
	return data

def read_mhd(filename):
    f = open(filename)
    info = {}
    line = f.readline().strip()
    while len(line) > 0:
        if not line.startswith('#'):
            pos = line.find('=')
            assert pos >= 0
            key = line[:pos].strip()
            val = line[pos+1:].strip()
            info[key] = val
        line = f.readline().strip()
	
    raw_filename = info['ElementDataFile']
    shape = info['DimSize'].split()
    shape[0],shape[1],shape[2] = int(shape[0]),int(shape[1]),int(shape[2])
    format = info['ElementType']
    format = 'int8'

    d = numpy.fromfile(filename[0:filename.rfind('/')+1]+raw_filename,dtype=format)
    d = numpy.reshape(d,shape)
    return d

def write_numpy(A, filename):
    savez(filename, {"data": A})

"""
def read_dicom(f):
    try:
        dcm = dicom.read_file(f, force=True)
    except:
        dcm = dicom.read_file(f)
        
    data = int16(dcm.PixelArray)

    if hasattr(dcm, 'RescaleSlope'):
        assert dcm.RescaleSlope == 1

    if hasattr(dcm, 'RescaleIntercept'):
        data += int16(dcm.RescaleIntercept)

    return data
"""

def read_2d_data(f):
    if f.endswith('.dat'): return read_dat(f)
    if f.endswith('.txt'): return loadtxt(f, dtype=float32)
#    if f.endswith('.dcm'): return read_dicom(f)
    if f.endswith('.mat'): return read_matlab(f)
    if f.endswith('.mtx'): return read_mtx(f)
    if f.endswith('.npz') or f.endswith('.npy'): return read_numpy(f)
    if any([f.endswith(e) for e in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP']]):
		img = Image.open(f)
		tmp = numpy.asarray(img)
		return tmp
    if f.endswith('.nrrd'): return read_nrrd(f)
    assert False, "unknown file type: " + f


def sort_numbered_files(files):  
    def n(f):
        dot_index = f.find('.')
        if dot_index >= 0: f = f[:dot_index]
        i = len(f) - 1
        while i >= 0 and f[i] >= '0' and f[i] <= '9': i -= 1
        return int(f[i+1:])

    def compare(x, y):
        return cmp(n(x), n(y))

    return sorted(files, compare)

"""
def sort_dicom_slices(files):
    "sort files by slice position"
    
    def slice_pos(f):
        slice = dicom.read_file(f, stop_before_pixels=True)
        return slice.SlicePosition
    
    try:
        pairs = [(f, slice_pos(f)) for f in files]
        pairs.sort(key = lambda x: x[1])
        return [f for (f, p) in pairs]
    except:
        return files
"""
def read_3d_data(f):
    if os.path.isdir(f):
        # read all files in directory as slices
        files = sort_numbered_files(os.listdir(f))
        
        # init numpy array for the voxel data
        img = read_2d_data(f + "/" + files[0])
        assert img.ndim == 2, files[0] + " should be a 2D slice"
        width, height = img.shape
        depth = len(files)
        data = zeros((width, height, depth), dtype = img.dtype, order = "F")
        data[:,:,0] = img

        # load voxel data
        for i, ff in enumerate(files):
            if i == 0: continue
            data[:,:,i] = read_2d_data(f + "/" + ff)

        return data        
    
#    if f.endswith('.dcm'): return dicomio.read_3d_dicom_data(f)
    if f.endswith('.mat'): return read_matlab(f)
    if f.endswith('.npz') or f.endswith('.npy'): return read_numpy(f)
    if f.endswith('.nrrd'): return read_nrrd(f)
    if f.endswith('.dat'): return read_dat(f)
    if f.endswith('.mhd'): return read_mhd(f)
    #assert False, "unknown file type: " + f


def read_4d_data(f):
    if os.path.isdir(f):
        # read all files in directory as slices
        files = sort_numbered_files(os.listdir(f))
        
        # init numpy array for the voxel data
        cube = read_3d_data(f + "/" + files[0])
        assert cube.ndim == 3, files[0] + " should be a 3D cube"
        width, height, depth = cube.shape
        frames = len(files)
        data = zeros((width, height, depth, frames), dtype = cube.dtype, order = "F")
        data[:,:,:,0] = cube
    
        # load voxel data
        for i, ff in enumerate(files):
            print 'loading cube:', ff
            if i == 0: continue
            data[:,:,:,i] = read_3d_data(f + "/" + ff)
    
        return data        

    if f.endswith('.mat'): return read_matlab(f)
    if f.endswith('.npz') or f.endswith('.npy'): return read_numpy(f)
    if f.endswith('.nrrd'): return read_nrrd(f)
    assert False, "unknown file type: " + f
    

