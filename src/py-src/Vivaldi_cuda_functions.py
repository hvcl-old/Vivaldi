import numpy
import math

# float2 functions
# ////////////////////////////////////////////////////////////
class float2(object):
	nbytes = 8
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __add__(self, other):
		if type(other) == float2: return float2(self.x + other.x, self.y + other.y)
		if type(other) == int: return float2(self.x + other, self.y + other)
		if type(other) == float: return float2(self.x + other, self.y + other)
		return self

	def __sub__(self, other):
		if type(other) == float2: return float2(self.x - other.x, self.y - other.y)
		if type(other) == int: return float2(self.x - other, self.y - other)
		if type(other) == float: return float2(self.x - other, self.y - other)
		return self

	def __mul__(self, other):
		if type(other) == float2: return float2(self.x * other.x, self.y * other.y)
		if type(other) == int: return float2(self.x * other, self.y * other)
		if type(other) == float: return float2(self.x * other, self.y * other)
		return self

	def __div__(self, other):
		if type(other) == float2: return float2(self.x / other.x, self.y / other.y)
		if type(other) == int: return float2(self.x / other, self.y / other)
		if type(other) == float: return float2(self.x / other, self.y / other)
		return self
	
	def __str__(self):
		return str([self.x, self.y])

	def as_array(self):
		return numpy.array([self.x, self.y], dtype=numpy.float32)

def make_float2(x, y=None):
	if x != None and y == None:
		return make_float2(x,y)
	return float2(x,y)


# float3 functions
# /////////////////////////////////////////////////////////////

#__add__ : Implements the plus "+" operator.
#__sub__ : Implements the minus "-" operator.
#__mul__ : Implements the multiplication "*" operator.
#__div__ : Implements the division "/" operator.

class float3(object):
	nbytes = 12
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

	def __add__(self, other):
		if type(other) == float3: return float3(self.x + other.x, self.y + other.y, self.z + other.z)
		if type(other) == int: return float3(self.x + other, self.y + other, self.z + other)
		if type(other) == float: return float3(self.x + other, self.y + other, self.z + other)
		return self

	def __sub__(self, other):
		if type(other) == float3: return float3(self.x - other.x, self.y - other.y, self.z - other.z)
		if type(other) == int: return float3(self.x - other, self.y - other, self.z - other)
		if type(other) == float: return float3(self.x - other, self.y - other, self.z - other)
		return self

	def __mul__(self, other):
		if type(other) == float3: return float3(self.x * other.x, self.y * other.y, self.z * other.z)
		if type(other) == int: return float3(self.x * other, self.y * other, self.z * other)
		if type(other) == float: return float3(self.x * other, self.y * other, self.z * other)
		return self

	def __div__(self, other):
		if type(other) == float3: return float3(self.x / other.x, self.y / other.y, self.z / other.z)
		if type(other) == int: return float3(self.x / other, self.y / other, self.z / other)
		if type(other) == float: return float3(self.x / other, self.y / other, self.z / other)
		return self
	def __str__(self):
		return str([self.x, self.y, self.z])

	def as_array(self):
		return numpy.array([self.x, self.y, self.z], dtype=numpy.float32)

def make_float3(x, y=None, z=None):
	if type(x) == float2:
		if y == None:return make_float3(x.x, x.y, 0)
		if y != None:return make_float3(x.x, x.y, y)	
	if type(x) == float4:
		return make_float3(x.x, x.y, x.z)
	if x != None and y == None and z == None: 
		return make_float3(x,x,x)

	return float3(x,y,z)

# float4 functions
# /////////////////////////////////////////////////////////////

#__add__ : Implements the plus "+" operator.
#__sub__ : Implements the minus "-" operator.
#__mul__ : Implements the multiplication "*" operator.
#__div__ : Implements the division "/" operator.

class float4(object):
	nbytes = 16
	def __init__(self, x, y, z, w):
		self.x = x
		self.y = y
		self.z = z
		self.w = w

	def __add__(self, other):
		if type(other) == float4: return float3(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)
		if type(other) == int: return float3(self.x + other, self.y + other, self.z + other, self.w + other)
		if type(other) == float: return float3(self.x + other, self.y + other, self.z + other, self.w + other)
		return self

	def __sub__(self, other):
		if type(other) == float4: return float3(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)
		if type(other) == int: return float3(self.x - other, self.y - other, self.z - other, self.w - other)
		if type(other) == float: return float3(self.x - other, self.y - other, self.z - other, self.w - other)
		return self

	def __mul__(self, other):
		if type(other) == float4: return float3(self.x * other.x, self.y * other.y, self.z * other.z, self.w * other.w)
		if type(other) == int: return float3(self.x * other, self.y * other, self.z * other, self.w * other)
		if type(other) == float: return float3(self.x * other, self.y * other, self.z * other, self.w * other)
		return self

	def __div__(slef, other):
		if type(other) == float4: return float3(self.x / other.x, self.y / other.y, self.z / other.z, self.w / other.w)
		if type(other) == int: return float3(self.x / other, self.y / other, self.z / other, self.w / other)
		if type(other) == float: return float3(self.x / other, self.y / other, self.z / other, self.w / other)
		return self
	
	def __str__(self):
		return str([self.x, self.y, self.z, self.w])


	def as_array(self):
		return numpy.array([self.x, self.y, self.z, self.w], dtype=numpy.float32)

def make_float4(x, y=None, z=None, w=None):

	if type(x) == float4:
		return make_float4(x.x, x.y, x.z, x.w)
	if type(x) == float3:
		if w == None: return make_float4(x.x, x.y, x.z, 0)
		if w != None: return make_float4(x.x, x.y, x.z, w)
	if x != None and y != None and z == None and w == None:
		return make_float4(x,x,x,y)
	if x != None and y == None and z == None and w == None:
		return make_float4(x,x,x,x)

	return float4(x,y,z,w)


# float functions
# /////////////////////////////////////////////////////////

def fminf(a, b):
	if type(a) == float4 and type(b) == float4:
		return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w))
	if type(a) == float3 and type(b) == float3:
		return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z))
	if type(a) == float2 and type(b) == float2:
		return make_float2(fminf(a.x,b.x), fminf(a.y,b.y))
	return min(a,b)

def fmaxf(a, b):
	if type(a) == float4 and type(b) == float4:
		return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w))
	if type(a) == float3 and type(b) == float3:
		return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z))
	if type(a) == float2 and type(b) == float2:
		return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y))
	return max(a,b)

def lerp(a, b, t):
	return a + t*(b-a)

def clamp(f, a, b):
	if type(f) == float2:
		if type(a) == float2:return make_float2(clamp(f.x,a.x,b.x),clamp(f.y,a.y,b.y))
		return make_float2(clamp(f.x,a,b),clamp(f.y,a,b))
	if type(f) == float3:
		if type(a) == float3:return make_float3(clamp(f.x,a.x,b.x),clamp(f.y,a.y,b.y),clamp(f.z,a.z,b.z))
		return make_float3(clamp(f.x,a,b),clamp(f.y,a,b),clamp(f.z,a,b))
	if type(f) == float4:
		if type(a) == float4:return make_float4(clamp(f.x,a.x,b.x),clamp(f.y,a.y,b.y),clamp(f.z,a.z,b.z),clamp(f.w,a.w,b.w))
		return make_float4(clamp(f.x,a,b),clamp(f.y,a,b),clamp(f.z,a,b),clamp(f.w,a,b))

	return fmaxf(a, fminf(f, b))

def step(edge, x):
	if x < edge: return 0
	return 1

def smoothstep(edge0, edge1, x):
	t = clamp((x-edge0)/(edge1-edge0), 0, 1)
	return t * t * (3 - 2*t)

def rect(edge0, edge1, x):
	if edge0 <= x and x <= edge1:return 1
	return 0

def dot(a, b):
	if type(a) == float2 and type(b) == float2:
		return a.x * b.x + a.y * b.y
	if type(a) == float3 and type(b) == float3:
		return a.x * b.x + a.y * b.y + a.z * b.z
	if type(a) == float4 and type(b) == float4:
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w

	# float
	return a * b

def cross(a, b):
	if type(a) == float3:
		return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
	pass

def length(v):
	import math
	return math.sqrt(dot(v,v))
	pass

def normalize(v):
	import math
	return v / math.sqrt(dot(v,v))
	pass

def floor(v):
	import math
	if type(v) == float2:
		return make_float2(math.floor(v.x), math.floor(v.y))
	if type(v) == float3:
		return make_float3(math.floor(v.x), math.floor(v.y), math.floor(v.z))

	# float
	return math.floor(v)
	pass

def ceil(v):
	import math
	if type(v) == float2:
		return make_float2(math.ceil(v.x), math.ceil(v.y))
	if type(v) == float3:
		return make_float3(math.ceil(v.x), math.ceil(v.y), math.ceil(v.z))
	
	# float
	return math.ceil(v)
	pass

def reflect(i, n):
	return i - 2.0 * n * dot(n, i)
	pass

def fabs(v):
	import math
	if type(v) == float2:
		return make_float2(math.fabs(v.x), math.fabs(v.y))
	if type(v) == float3:
		return make_float3(math.fabs(v.x), math.fabs(v.y), math.fabs(v.z))
	
	# float
	return math.fabs(v)
	pass


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
	if type(x) == float2:
		y = x.y
		x = x.x
	size = image.shape
	
	fx = floor(x)
	fy = floor(y)
	cx = ceil(x)
	cy = ceil(y)
	dx = x - fx
	dy = y - fy


	q00 = 0
	q01 = 0
	q10 = 0
	q11 = 0

	if 0 <= fx and fx < size[0] and 0 <= fy and fy < size[1]: q00 = image[fx][fy]
	if 0 <= cx and cx < size[0] and 0 <= fy and fy < size[1]: q10 = image[cx][fy]
	if 0 <= fx and fx < size[0] and 0 <= cy and cy < size[1]: q01 = image[fx][cy]
	if 0 <= cx and cx < size[0] and 0 <= cy and cy < size[1]: q11 = image[cx][cy]

	# lerp along y
	q0 = lerp(q00, q01, dy)
	q1 = lerp(q10, q11, dy)

	# lerp along x
	q = lerp(q0, q1, dx)
	return q

def linear_gradient_2d(volume, p, t):
	if p != float2:
		p = make_float2(p, t)
	delta = 0.0001
	dx = linear_query_2d(volume, make_float2(p.x + delta, p.y)) - linear_query_2d(volume, make_float2(p.x - delta, p.y))
	dy = linear_query_2d(volume, make_float2(p.x, p.y + delta)) - linear_query_2d(volume, make_float2(p.x, p.y - delta))
	return make_float2(dx, dy) / (2 * delta);

def linear_query_2d_rgb(image, x, y=None):
	if type(x) == float2:
		y = x.y
		x = x.x
	size = image.shape

	fx = floor(x)
	fy = floor(y)
	cx = ceil(x)
	cy = ceil(y)

	dx = x - fx
	dy = y - fy

	n = len(image.shape)
	q00 = [0]*n
	q10 = [0]*n
	q01 = [0]*n
	q11 = [0]*n
	
	q0 = [0]*n
	q1 = [0]*n

	q = [0]*n
	for i in range(n):
		if 0 <= fx and fx < size[0] and 0 <= fy and fy < size[1]: q00[i] = image[fx][fy][i]
		else: q00[i] = 0
		if 0 <= cx and cx < size[0] and 0 <= fy and fy < size[1]: q10[i] = image[cx][fy][i]
		else: q10[i] = 0
		if 0 <= fx and fx < size[0] and 0 <= cy and cy < size[1]: q01[i] = image[fx][cy][i]
		else: q01[i] = 0
		if 0 <= cx and cx < size[0] and 0 <= cy and cy < size[1]: q11[i] = image[cx][cy][i]
		else: q11[i] = 0
	
		# lerp along y
		q0[i] = lerp(q00[i], q01[i], dy)
		q1[i] = lerp(q10[i], q11[i], dy)

		# lerp along x
		q[i] = lerp(q0[i], q1[i], dx)

	return make_float3(q[0], q[1], q[2])

def linear_gradient_2d_rgb(image, p, t):
	if p != float2:
		p = make_float2(p, t)
	delta = 0.0001
	dx = linear_query_2d_rgb(image, make_float2(p.x + delta, p.y)) - linear_query_2d_rgb(image, make_float2(p.x - delta, p.y))
	dy = linear_query_2d_rgb(image, make_float2(p.x, p.y + delta)) - linear_query_2d_rgb(image, make_float2(p.x, p.y - delta))
	return make_float2(length(dx), length(dy)) / (2 * delta);


# 3D data query functions
########################################################################################

def float2_to_int2(a):
	if type(a) == float2:
		return make_int2(int(a.x), int(a.y))
	pass

def float3_to_int3(a):
	if type(a) == float3:
		return make_int3(int(a.x), int(a.y), int(a.z))
	pass

def cubic_query_3d(volume,x, y=None, z=None):
	if type(x) == float3:
		z = x.z
		y = x.y
		x = x.x
	size = volume.shape

	fx = floor(x)
	fy = floor(y)
	fz = floor(z)
	cx = ceil(x)
	cy = ceil(y)
	cz = ceil(z)

	if fx > size[0]-1:return 0
	if fy > size[1]-1:return 0
	if fz > size[2]-1:return 0
	if fx < 0: return 0
	if fy < 0: return 0
	if fz < 0: return 0

	if cx > size[0]-1:return 0
	if cy > size[1]-1:return 0
	if cz > size[2]-1:return 0
	if cx < 0: return 0
	if cy < 0: return 0
	if cz < 0: return 0

	dx = x - fx
	dy = y - fy
	dz = z - fz

	q000 = volume[fx][fy][fz]
	q100 = volume[cx][fy][fz]
	q010 = volume[fx][cy][fz]
	q110 = volume[cx][cy][fz]
	q001 = volume[fx][fy][cz]
	q101 = volume[cx][fy][cz]
	q011 = volume[fx][cy][cz]
	q111 = volume[cx][cy][cz]

	# lerp along z
	q00 = lerp(q000, q001, dz)
	q01 = lerp(q010, q011, dz)
	q10 = lerp(q100, q101, dz)
	q11 = lerp(q110, q111, dz)

	# lerp along y
	q0 = lerp(q00, q01, dy)
	q1 = lerp(q10, q11, dy)

	# lerp along x
	q = lerp(q0, q1, dx)
	return q

def linear_query_3d(volume, x, y=None, z=None):
	if type(x) == float3:
		z = x.z
		y = x.y
		x = x.x
	return cubic_query_3d(volume, x, y, z)

def linear_gradient_3d(volume, p):
	delta = 0.0001
	dx = linear_query_3d(volume, make_float3(p.x + delta, p.y, p.z)) - linear_query_3d(volume, make_float3(p.x - delta, p.y, p.z))
	dy = linear_query_3d(volume, make_float3(p.x, p.y + delta, p.z)) - linear_query_3d(volume, make_float3(p.x, p.y - delta, p.z))
	dz = linear_query_3d(volume, make_float3(p.x, p.y, p.z + delta)) - linear_query_3d(volume, make_float3(p.x, p.y, p.z - delta))
	return make_float3(dx, dy, dz) / (2 * delta);

def cubic_gradient_3d(volume, p):
	return linear_gradient_3d(volume, p)

#rotation functions
########################################################################################

# Rgba
########################################################################################
class Rgba(object):
	def __init__(self, r=None, g=None, b=None, a=None):
		if r == None and g == None and b == None and a == None:
			self.r = 0
			self.g = 0
			self.b = 0
			self.a = 0
		elif r != None and g != None and b != None and a != None:
			self.r = clamp(r * 255, 0.0, 255.0)
			self.g = clamp(g * 255, 0.0, 255.0)
			self.b = clamp(b * 255, 0.0, 255.0)
			self.a = clamp(a, 0.0, 1.0)
		elif type(r) == float4:
			self.r = clamp(r.x*255, 0, 255.0)
			self.g = clamp(r.y*255, 0, 255.0)
			self.b = clamp(r.z*255, 0, 255.0)
			self.a = clamp(r.w, 0, 1.0)
		elif type(r) == float3 and a == None:
			self.r = clamp(r.x*255, 0, 255.0)
			self.g = clamp(r.y*255, 0, 255.0)
			self.b = clamp(r.z*255, 0, 255.0)
			self.a = 0
	def __str__(self):
		return str([self.r, self.g, self.b, self.a])

	def as_array(self):
		return numpy.array([self.r, self.g, self.b, self.a], dtype=numpy.float32)

# Rgb 
########################################################################################
class Rgb(object):
	def __init__(self, r=None, g=None, b=None):
		if r == None and g == None and b == None:
			self.r = 0
			self.g = 0
			self.b = 0
		elif r != None and g != None and b != None:
			self.r = clamp(r * 255, 0.0, 255.0)
			self.g = clamp(g * 255, 0.0, 255.0)
			self.b = clamp(b * 255, 0.0, 255.0)
		elif type(r) == float3 and g == None and b == None:
			self.r = clamp(r.x*255, 0, 255.0)
			self.g = clamp(r.y*255, 0, 255.0)
			self.b = clamp(r.z*255, 0, 255.0)
		elif type(r) == float4 and g == None and b == None:
			self.r = clamp(r.x*255, 0, 255.0)
			self.g = clamp(r.y*255, 0, 255.0)
			self.b = clamp(r.z*255, 0, 255.0)
	def __str__(self):
		return str( [self.r, self.g, self.b] )

	def as_array(self):
		return numpy.array([self.r, self.g, self.b], dtype=numpy.float32)
"""
def line_iter(S, E, d):
	rb = []
	step = normalize(E-S)*d
	len = length(E-S)
	P = S
	if S.x == E.x and S.y == E.y and S.z == E.z: return make_float3(0,0,1)

	
	for i in range(0,step+1):
		P = S + (E-S)*i/step
		rb.append(P)
	return rb
"""
class line_iter:
	def __init__(self, start, end, distance):
		self.start = start
		self.end = end
		self.distance = distance
		self.differ = length(end-start)
		self.direction = normalize(end-start)
		self.current = start
		self.step = self.direction * self.distance

	def __iter__(self):
		return self

	def next(self):
		D = length(self.current-self.start)
		if D > self.differ:
			 raise StopIteration
		else:
			self.current += self.step
			return self.current-self.step


