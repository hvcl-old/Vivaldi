---------------------------------------------------------------------------------------------------------------------------
main:
This is start python code for UeberShadie. main count number of CPU and GPU of
each nodes and make 
'Vivaldi_main_manager' in the head
'Vivaldi_memory_manager' in the head
'Vivaldi_scheduler' in the head
'CPU_unit' in the nodes
'GPU_unit' in the nodes

This is originally designed for communicated to client using TCP/IP socket and
make a main manager for each clients. but currently not doing socket work and
only manager one work.
 
Function List
def make_mpi_spawn_info()
def run_main_manager()
def run_memory_manager()
def run_scheduler()
def make_computing_unit_list()
def clean_computing_unit()
def get_CPU_execid_list()
def get_GPU_execid_list()
def deploy_CPU_function(dst, code)
def deploy_GPU_function(dst, code)
def divide_args(args)
def strip_func(line)
def register_comm(new_comm)


---------------------------------------------------------------------------------------------------------------------------
misc.py
small helper functions go here

Function List
def read_file(name)
def write_file(name, x)
def normalize(v)
def cross(a, b)
def mk_frame(position, target, up)
def find_index(p, l)
def rotation_matrix(axis, angle)


---------------------------------------------------------------------------------------------------------------------------
matrixio.py   
Read functions are here

Function List
def read_mtx(filename)
def write_mtx(A, filename)
def read_matlab(filename)
def write_matlab(A, filename)
def read_numpy(filename)
def read_nrrd(filename)
def read_dat(filename)
def read_mhd(filename)
def write_numpy(A, filename)
def read_dicom(f)
def read_2d_data(f)
def sort_numberred_files(files)
def sort_dicom_slices(files)
def read_3d_data(f)
def read_4d_data(f)



---------------------------------------------------------------------------------------------------------------------------
CPU_unit.py  
CPU management code are here

Function List
def request_data_to_CPU_unit(data_name, dst)
def request_data_to_GPU_unit(data_name, dst)
def notice(data_name, data_package, reset)
def ask_about_var(name)
def send(data_name, target)
def save_data(data_name, data)

---------------------------------------------------------------------------------------------------------------------------
GPU_unit.py
GPU management code are here using pycuda

Function List
def idle()
def notice(data_package)
def send(unique_id, decomposition_id, target)
def recv()
def save_data(data, data_package)
def create_helper_textures()
def update_random_texture(random_texture)


---------------------------------------------------------------------------------------------------------------------------
Vivaldi_functions.py
VIVALDI compiler are here. There are 4 code management classes, one for
basic code management, three for each Vivaldi, python and cuda code management.
code_manager is for basic code management class. like divide python like
programming code to several parts and  basic getter and setters and common
functions for specific code managers. 


class code_manager(object)
Function List
def __init__(self, code)
def parse_head(self)
def parse_name(self)
def parse_args(self)
def parse_body(self)
def add_volume(self, volume)
def make_volume_list(self)
def add_type(self, var, dtype)
def get_type(self, var)
def add_setter(self, left, right, dtype)
def find_divide(self, line)
def find_modifier(self, left, right)
def parse_modifier(self, left, right, leftmost_var)
def divide_args(self, args)
def strip_func(self, line)
def make_func_list(self)
def parse_function(self, func_name, args_list, dtype_list)
def line_parser(self, line, leftmost, leftmost_var)
def get_args(self)
def set_args(self, args)
def get_name(self)
def set_name(self, name)
def get_body(self)
def set_body(self, body)
def get_head(self)
def set_head(self, head)
def get_code(self)
def get_volume_list(self)
def set_volume_list(self, volume_list)
def get_split_arg_list(self)
def set_split_arg_list(self, split_arg_list)
def get_return_dtype(self)
def set_return_dtype(self, dtype)
def __str__(self)
	
class Vivaldi_code_manager(code_manager)
Function List
def __init__(self, code)
def parse_Vivaldi(self, Vivaldi_code)

class python_code_manager(code_manager)
Function List
def __init__(self, code, name=None, execid=[], work_range=[], split=[], parallel=[]):
def copy(self)
def parallelize(self, args=[], dtype=None, work_range=[])
def manage_print_return(self, line)
def parse_save_image(self, code)
def sh2py(self)
def set_name(self, name)
def set_args(self, args)
def add_tab(self, tab_count)
def set_body(self, body)
def get_python_code(self)
def set_python_code(self, python_code)
def get_Vivaldi_code(self)
def set_Vivaldi_code(self, Vivaldi_code)
def make_code(self)
def change_return(self, range_arg_list=[])
def get_code(self)
def set_code(self, Vivaldi_code)

class cuda_code_manager(code_manager)
Function List
def __init__(self, code, name, args_in=[], work_range=[], split=[], parallel=[])
def manage_print_return(self, line)
def find_divide(self, line)
def parse_volume(self, line)
def sh2cu_args(self, func_args)
def parse_if(self, code)
def parse_line(self, code)
def parse_for(self, code)
def parse_while(self, code)
def attach_semicolon(self, code)
def sh2cu_body(self, func_body, func_args)
def sh2cu_range_check(self, args)
def sh2cu(self)
def get_attachment(self)
def get_core_code(self)

class Vivaldi_function(object)
Function List
def __init__(self, Vivaldi_code)
def get_name(self)
def get_args(self)
def get_Vivaldi_code(self)
def set_shdie_code(self, code)
def get_python_code(self, name, args, work_range)
def set_python_code(self, code)
def get_attachment(self)
def get_cuda_core_code(self, name)
def get_cuda_code(self, name, args=[], work_range=[], split=[])
def get_code(self, language)
def get_Vivaldi_code_manager(self)
def get_python_code_manager(self)
def get_cuda_code_manager(self, name)
def get_parallel_python_code_manager(self, name)

class Vivaldi_functions
Function_list
def __init__(self, Vivaldi_code)
def keys(self)
def get_Vivaldi_code(self, name)
def get_python_code(self, name, args=[], work_range=[])
def get_cuda_code(self)
def get_cuda_core_code(self, name, args=[], work_range=[], split=[])
def get_attachment(self)
def get_python_code_manager(self, name, parallel_name=None)
def get_cuda_code_manager(self, name, parallel_name)
def get_code(self, name, language)
def get_function(self, name)
def get_front(self)

---------------------------------------------------------------------------------------------------------------------------
Vivaldi_main_manager.py       

Function List
def LoadMatrx(name)
def LoadIdentity()
def Rotate(angle, x, y, z)
def Translate(x, y, z)
def Scaled(x, y, z)
def Merge(data, f, m)
def run_function(function_name, arg_list=[], return_name=None, execid=[], work_range=[], split_list=[])
	def make_tasks(arg_packages, i, work_range, full_work_range, sg, function_package)
def mem_unique_id(unique_id)
def mem_number(unique_id, number_of_segments)
def notice(data_package=[])
def merge_data(old_id, new_id, f=None, method=None)
def send_data_to_CPU_unit(dst, data_name, data)
def request_data_to_CPU_unit(data_name, dst)
def send_data_to_GPU_unit(dst, data, data_package)
def request_data_package(unique_id, decomposition_id, dst)
def request_data(u, d)
	def request_data_to_GPU_unit(u,d)
def	register_function(execid, function_package)
def isCPU(execid)
def isGPU(execid)
def get_any_CPU()
def get_any_GPU()
def get_another_CPU(execid)
def get_another_GPU(execid)
def get_CPU_list(number)
def get_CPU_list(number)
def get_dtype_shape(dtype=None)
def save_image_inner(data_name=None, buf=None, dtype=None, extension=None)
def save_image(data_name, extension='png')
def divide_args(args)
def strip_func(line)



---------------------------------------------------------------------------------------------------------------------------
Vivaldi_memory.py

Vriables
	self.unique_id
	self.decomposition_id
	self.data_name
	self.shape
	self.memory_shape
	self.dtype
	self.contents_dtype
	self.contents_memory_dtype

Function List
def __init__(self)
def __str__(self)

---------------------------------------------------------------------------------------------------------------------------
Vivaldi_cuda_attachment.cu  

Variable
#define GPU inline __device__

__device__ int execid
__device__ float modelview[4][4]
__device__ inv_modelview[4][4]

Function List
// float functions
////////////////////////////////////////////////////////////////////////////////
GPU float lerp(float a, float b, float t)
GPU float clamp(float f, float a, float b)
GPU float step(float edge, float x)
GPU float smoothstep(float edge0, float edge1, float x)
GPU float rect(float edge0, float edge1, float x)

// float2 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
GPU float2 make_float2(float s)
GPU float2 make_float2(int2 a)
// negate
GPU float2 operator-(float2 a)
// addition
GPU float2 operator+(float2 a, float2 b)
GPU void operator+=(float2 &a, float2 b)
// subtract
GPU float2 operator-(float2 a, float2 b)
GPU void operator-=(float2 &a, float2 b)
// multiply
GPU float2 operator*(float2 a, float2 b)
GPU float2 operator*(float2 a, float s)
GPU float2 operator*(float s, float2 a)
GPU void operator*=(float2 &a, float s)
// divide
GPU float2 operator/(float2 a, float2 b)
GPU float2 operator/(float2 a, float s)
GPU float2 operator/(float s, float2 a)
GPU void operator/=(float2 &a, float s)
// lerp
GPU float2 lerp(float2 a, float2 b, float t)
// clamp
GPU float2 clamp(float2 v, float a, float b)
GPU float2 clamp(float2 v, float2 a, float2 b)
// dot product
GPU float dot(float2 a, float2 b)
// length
GPU float length(float2 v)
// normalize
GPU float2 normalize(float2 v)
// floor
GPU float2 floor(const float2 v)
// reflect
GPU float2 reflect(float2 i, float2 n)
// absolute value
GPU float2 fabs(float2 v)
// float3 functions
////////////////////////////////////////////////////////////////////////////////
// additional constructors
GPU float3 make_float3(float s)
GPU float3 make_float3(float2 a)
GPU float3 make_float3(float2 a, float s)
GPU float3 make_float3(float4 a)
GPU float3 make_float3(int3 a)
// negate
GPU float3 operator-(float3 a)
// min
GPU float3 fminf(float3 a, float3 b)
// max
GPU float3 fmaxf(float3 a, float3 b)
// addition
GPU float3 operator+(float3 a, float3 b)
GPU float3 operator+(float3 a, float b)
GPU float3 operator+(float a, float3 b)
GPU void operator+=(float3 &a, float3 b)
// subtract
GPU float3 operator-(float3 a, float3 b)
GPU float3 operator-(float3 a, float b)
GPU void operator-=(float3 &a, float3 b)
// multiply
GPU float3 operator*(float3 a, float3 b)
GPU float3 operator*(float3 a, float s)
GPU float3 operator*(float s, float3 a)
GPU void operator*=(float3 &a, float s)
// divide
GPU float3 operator/(float3 a, float3 b)
GPU float3 operator/(float3 a, float s)
GPU float3 operator/(float s, float3 a)
GPU void operator/=(float3 &a, float s)
// lerp
GPU float3 lerp(float3 a, float3 b, float t)
// clamp
GPU float3 clamp(float3 v, float a, float b)
GPU float3 clamp(float3 v, float3 a, float3 b)
// dot product
GPU float dot(float3 a, float3 b)
// cross product
GPU float3 cross(float3 a, float3 b)
// length
GPU float length(float3 v)
// normalize
GPU float3 normalize(float3 v)
// floor
GPU float3 floor(const float3 v)
// reflect
GPU float3 reflect(float3 i, float3 n)
// absolute value
GPU float3 fabs(float3 v)
// float4 functions
////////////////////////////////////////////////////////////////////////////////
// additional constructors
GPU float4 make_float4(float s)
GPU float4 make_float4(float a, float s)
GPU float4 make_float4(float3 a)
GPU float4 make_float4(float3 a, float w)
GPU float4 make_float4(int4 a)
// negate
GPU float4 operator-(float4 a)
// min
GPU float4 fminf(float4 a, float4 b)
// max
GPU float4 fmaxf(float4 a, float4 b)
// addition
GPU float4 operator+(float4 a, float4 b)
GPU void operator+=(float4 &a, float4 b)
// subtract
GPU float4 operator-(float4 a, float4 b)
GPU void operator-=(float4 &a, float4 b)
// multiply
GPU float4 operator*(float4 a, float s)
GPU float4 operator*(float s, float4 a)
GPU void operator*=(float4 &a, float s)
// divide
GPU float4 operator/(float4 a, float4 b)
GPU float4 operator/(float4 a, float s)
GPU float4 operator/(float s, float4 a)
GPU void operator/=(float4 &a, float s)
// lerp
GPU float4 lerp(float4 a, float4 b, float t)
// clamp
GPU float4 clamp(float4 v, float a, float b)
GPU float4 clamp(float4 v, float4 a, float4 b)
// dot product
GPU float dot(float4 a, float4 b)
// length
GPU float length(float4 r)
// normalize
GPU float4 normalize(float4 v)
// floor
GPU float4 floor(const float4 v)
// absolute value
GPU float4 fabs(float4 v)
// Frame
////////////////////////////////////////////////////////////////////////////////
{
public:
    float3 x, y, z, origin;
    GPU void setDefault(float3 position)
    GPU void lookAt(float3 position, float3 target, float3 up)
    GPU float3 getVectorToWorld(float3 v)
    GPU float3 getPointToWorld(float3 p)
};
// transfer2
////////////////////////////////////////////////////////////////////////////////
GPU float transfer2(
    float x0, float f0,
    float x1, float f1,
    float x)
GPU float2 transfer2(
    float x0, float2 f0,
    float x1, float2 f1,
    float x)
GPU float3 transfer2(
    float x0, float3 f0,
    float x1, float3 f1,
    float x)
GPU float4 transfer2(
    float x0, float4 f0,
    float x1, float4 f1,
    float x)
// transfer3
////////////////////////////////////////////////////////////////////////////////
GPU float transfer3(
    float x0, float f0,
    float x1, float f1,
    float x2, float f2,
    float x)
GPU float2 transfer3(
    float x0, float2 f0,
    float x1, float2 f1,
    float x2, float2 f2,
    float x)
GPU float3 transfer3(
    float x0, float3 f0,
    float x1, float3 f1,
    float x2, float3 f2,
    float x)
GPU float4 transfer3(
    float x0, float4 f0,
    float x1, float4 f1,
    float x2, float4 f2,
    float x)
// helper textures for cubic interpolation and random numbers
////////////////////////////////////////////////////////////////////////////////
texture<float4, 2, cudaReadModeElementType> hgTexture;
texture<float4, 2, cudaReadModeElementType> dhgTexture;
texture<int, 2, cudaReadModeElementType> randomTexture;
GPU float3 hg(float a)
GPU float3 dhg(float a)
//iterators

class make_line_iter
{
public:
    float3 S,E,P,step;
    float len;
    GPU make_line_iter(float3 from, float3 to, float d)
    GPU float3 begin()
    GPU bool hasNext()
    GPU float3 next()
    GPU float3 direction()
};

class make_plane_iter
{
public:
    float2 S;
    float d;
    int max_step, step;
    int width;
    float x,y;
    GPU make_plane_iter(float2 point, float size)
    GPU make_plane_iter(int x, int y, float size)
    GPU make_plane_iter(float x, float y, float size)
    GPU float2 begin()
    GPU bool hasNext()
    GPU float2 next()
};
class make_cube_iter
{
public:
    float3 S;
    int d;
    int width;
    int max_step, step;
    float x,y,z;
    GPU make_cube_iter(float3 point, float size)
    GPU make_cube_iter(int x,int y,int z, float size)
    GPU make_cube_iter(float x, float y, float z, float size)
    GPU float3 begin()
    GPU bool hasNext()
    GPU float3 next()
};
// data query functions
//////////////////////////////////////////////////////////////////////////////
#define INF            __int_as_float(0x7f800000)
GPU int2 float2_to_int2(float2 a)
GPU int3 float3_to_int3(float3 a)
// 2D data query functions
////////////////////////////////////////////////////////////////////////////////
GPU float linear_query_2d(float* image, float2 p ,int2 size)
GPU float linear_query_2d(float* image, float x, float y, int2 size)
GPU float2 linear_gradient_2d(float* image, float2 p, int2 size)
GPU float2 linear_gradient_2d(float* image, float x, float y, int2 size)
GPU float2 linear_gradient_2d(float* image, int x, int y, int2 size)
GPU float3 linear_query_2d_rgb(float* image, float2 p, int2 start, int2 size)
GPU float3 linear_query_2d_rgb(float* image, float x, float y, int2 start, int2 size)
GPU float3 linear_query_2d_rgb(float* image, int x, int y, int2 start, int2 size)
GPU float2 linear_gradient_2d_rgb(float* image, float2 p, int2 start, int2 size)
GPU float2 linear_gradient_2d_rgb(float* image, float x, float y, int2 start, int2 size)
GPU float2 linear_gradient_2d_rgb(float* image, int x, int y, int2 start, int2 size)
GPU float query_2d(float* image, float2 p, int2 start, int2 end)
GPU float query_2d(float* image, float x, float y, int2 start, int2 end)
GPU float query_2d(float* image, int x, int y, int2 start, int2 end)
GPU float3 query_2d_rgb(float* image, float2 p, int2 start, int2 end)
GPU float3 query_2d_rgb(float* image, float y, float x, int2 start, int2 end)
GPU float3 query_2d_rgb(float* image, int x, int y, int2 start, int2 end)
GPU float d_lerp(float a, float b, float t)
// 3D data query functions
////////////////////////////////////////////////////////////////////////////////
GPU float linear_query_3d(float* volume, float3 p, int3 start, int3 end)
GPU float linear_query_3d(float* volume, float x, float y, float z, int3 start, int3 end)
GPU float3 linear_gradient_3d(float* volume, float3 p, int3 start, int3 end)
GPU float linear_query_1d(float* line, float x, int size)
GPU float4 linear_query_1d_rgba(float* line, float x, int size)
GPU float cubic_query_3d(float* volume, float3 p, int3 start, int3 end)
GPU float3 cubic_gradient_3d(float* data, float3 p, int3 start, int3 end)
//rotate functions
///////////////////////////////////////////////////////////////////////////////////
GPU float arccos(float angle)
GPU float arcsin(float angle)
GPU float norm(float3 a)
GPU float2 intersectSlab(float p, float d, float2 slab)
GPU float2 intersectIntervals(float2 a, float2 b)
GPU float2 intersectCube(float3 rayOrigin, float3 rayDirection,
                         float2 x_range_to_draw,  float2 y_range_to_draw,float2 z_range_to_draw)
GPU make_line_iter perspective_iter(float *volume, float x, float y, float step, float near, int3 start, int3 end)
GPU make_line_iter orthogonal_iter(float *volume, float2 p, float step, int3 start, int3 end)
GPU make_line_iter orthogonal_iter(float *volume, float x, float y, float step, int3 start, int3 end)
// rgba class
class Rgba
{
public:
    float r, g, b, a;
    GPU Rgba(float3 rgb, float a_in)
    GPU Rgba(float4 rgba)
    GPU Rgba(float r_in, float g_in, float b_in, float a_in)
    GPU Rgba(float c)
    GPU Rgba()
};
// rgb class
class Rgb
{
public:
    float r, g, b;
    GPU Rgb(float3 rgb)
    GPU Rgb(float4 rgb)
    GPU Rgb(float r_in, float g_in, float b_in)
    GPU Rgb()
};

GPU float3 phong(float3 L, float3 N, float3 omega, float3 kd, float3 ks, float n, float3 amb)
GPU float3 diffuse(float3 L, float3 N, float3 kd)
template<typename R,typename T> GPU R laplacian(T* image, float2 p, SHADIE_DATA_RANGE* sdr)
template<typename R,typename T> GPU R laplacian(T* image, float x, float y, SHADIE_DATA_RANGE* sdr)

---------------------------------------------------------------------------------------------------------------------------
Vivaldi_memory_manager.py     

Function List
def isCPU(execid):
def isGPU(execid):
def run_CPU_function(dst, function_package):
def run_GPU_function(dst, function_package):
def func_compositing(old_id, new_id, func_name, method):
	def merge(ou, od, u1, d1, u2, d2):
def alpha_compositing(old_id, new_id):
    def merge1(ou, od, u1, d1, u2, d2):
def add_compositing(old_id, new_id):
    def merge2(ou, od, u1, d1, u2, d2):
def screen_compositing(old_id, new_id):
    def merge2(ou, od, u1, d1, u2, d2):
def launch_function():
def send_order(f, t, u, d):
def register_function(function_package):


---------------------------------------------------------------------------------------------------------------------------
Vivaldi_scheduler.py

Function List


---------------------------------------------------------------------------------------------------------------------------
Vivaldi_cuda_functions.py   

Function List
# float2 functions
# ////////////////////////////////////////////////////////////
class float2(object):
    nbytes = 8
    def __init__(self, x, y):
    def __add__(self, other):
    def __sub__(self, other):
    def __mul__(self, other):
    def __div__(self, other):
    def __str__(self):
    def as_array(self):
def make_float2(x, y=None):
# float3 functions
# /////////////////////////////////////////////////////////////
#__add__ : Implements the plus "+" operator.
#__sub__ : Implements the minus "-" operator.
#__mul__ : Implements the multiplication "*" operator.
#__div__ : Implements the division "/" operator.
class float3(object):
    nbytes = 12
    def __init__(self, x, y, z):
    def __add__(self, other):
    def __sub__(self, other):
    def __mul__(self, other):
    def __div__(self, other):
    def __str__(self):
    def as_array(self):
def make_float3(x, y=None, z=None):
# float4 functions
# /////////////////////////////////////////////////////////////
#__add__ : Implements the plus "+" operator.
#__sub__ : Implements the minus "-" operator.
#__mul__ : Implements the multiplication "*" operator.
#__div__ : Implements the division "/" operator.
class float4(object):
    nbytes = 16
    def __init__(self, x, y, z, w):
    def __add__(self, other):
    def __sub__(self, other):
    def __mul__(self, other):
    def __div__(slef, other):
    def __str__(self):
    def as_array(self):
def make_float4(x, y=None, z=None, w=None):
# float functions
# /////////////////////////////////////////////////////////
def fminf(a, b):
def fmaxf(a, b):
def lerp(a, b, t):
def clamp(f, a, b):
def step(edge, x):
def smoothstep(edge0, edge1, x):
def rect(edge0, edge1, x):
def dot(a, b):
def cross(a, b):
def length(v):
def normalize(v):
def floor(v):
def ceil(v):
def reflect(i, n):
def fabs(v):
# Frame
########################################################################################
# transfer2
########################################################################################
# transfer3
########################################################################################
# helper texture for cubic interpolation and random numbers
########################################################################################
#iterators
########################################################################################
# 2D data query functions
########################################################################################
def linear_query_2d(image, x, y=None):
def linear_gradient_2d(volume, p, t):
def linear_query_2d_rgb(image, x, y=None):
def linear_gradient_2d_rgb(image, p, t):
# 3D data query functions
########################################################################################
def float2_to_int2(a):
def float3_to_int3(a):
def cubic_query_3d(volume,x, y=None, z=None):
def linear_query_3d(volume, x, y=None, z=None):
def linear_gradient_3d(volume, p):
def cubic_gradient_3d(volume, p):
#rotation functions
########################################################################################
# Rgba
########################################################################################
class Rgba(object):
    def __init__(self, r=None, g=None, b=None, a=None):
    def __str__(self):
    def as_array(self):
# Rgb 
########################################################################################
class Rgb(object):
    def __init__(self, r=None, g=None, b=None):
    def __str__(self):
    def as_array(self):
class make_line_iter:
    def __init__(self, start, end, distance):
    def __iter__(self):
    def next(self):


---------------------------------------------------------------------------------------------------------------------------
Vivaldi_load.py

Function List
def load_data_2d(filename, dtype=None):
def load_data_1d_rgb(filename, dtype=None):
def load_data_1d_rgba(filename, dtype=None):
def load_data_2d_rgb(filename, dtype=None):
def load_data_2d_rgba(filename, dtype):
def load_data_3d(filename, dtype=None, access_type=None):
def load_data_4d(filename, dtype):


---------------------------------------------------------------------------------------------------------------------------
Vivaldi_memory_packages.py    

Function List
class Function_package():
    def __init__(self):
    def __str__(self):
    def copy(self):
class Data_package():
    def __init__(self, u = None, d = None, m = None):
    def __str__(self):
    def copy(self):


---------------------------------------------------------------------------------------------------------------------------
texture.py

Function List
def create_2d_texture(a, module, variable, point_sampling=False):
def create_2d_rgba_texture(a, module, variable, point_sampling=False):
def create_3d_texture(a, module, variable, point_sampling=False):
def create_4d_texture(a, module, variable, point_sampling=False):
def update_2d_texture(texref, newdata):
def load_data_2d(filename, name, mod):
def load_data_2d_rgba(filename, name, mod):
def load_data_3d(filename, name, mod, rank, data_divide):
def load_data_4d(filename, name, mod):



---------------------------------------------------------------------------------------------------------------------------
asdf.cu      
cuda code for compile


---------------------------------------------------------------------------------------------------------------------------
result                     
save_image results are saved here

---------------------------------------------------------------------------------------------------------------------------
Vivaldi                  
start script to VIVALDI usage is 
e.x) Vivaldi ../example/edge_detection_flower.uVivaldi

---------------------------------------------------------------------------------------------------------------------------
log 
VIVALDI execution log
 
