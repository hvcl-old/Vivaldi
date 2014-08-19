#1. START VIVALDI
###HELLO WORLD
 The basic structure of a Vivaldi is a collection of functions. Vivaldi function is defined by the **def** identifier as in python.  There are two types of functions the main function (**def main()**) and worker functions (e.g., **def function()**).

**Example**
```bash
vi helloworld.vvl
```
```python
def function():
        return "Hello, world!"

def main():
        result = function()
        print result
```
**Result**
```
Hello, world!
```

#2. INPUT AND OUTPUT
###Load input
 DATA_PATH is set as Vivaldi/data/.

- data = load_data_2d(DATA_PATH + ‘file_name’)
- data = load_data_2d(DATA_PATH + ‘file_name’, data_type)
- data = load_data_3d(DATA_PATH + ‘file_name’)
- data = load_data_2d(DATA_PATH + ‘file_name’, data_type)

###Save output
 The folder named result is automatically created and file name which hasn’t file name is automatically saved as ordered number.

- save_image(result)
- save_image(result, ‘file_name’)

**Example**
```python
def half(image, x, y):
	#size of image is 1280x853
    a = point_query_2d(image, x, y)
    ret = make_float3(0)
    #red
    if x<=400:
        ret = make_float3(a.x, 0, 0)
    #green
    elif x<=400 and y>200:
        ret = make_float3(0, a.y, 0)
    #red+blue
    elif x>400 and y<=200:
        ret = make_float3(a.x, 0, a.z)
	#green+blue
    else:
        ret = make_float3(0, a.y, a.z)
    return ret

def main():
    image = load_data_2d(DATA_PATH + 'image.jpg')
	output = half(image, x, y).range(x=0:1280, y=0:853)
	                          .dtype(image, uchar)
    save_image(output,'color_test.png')
```
**Original**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/aa.jpg?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9hYS5qcGciLCJleHBpcmVzIjoxNDA5MjEwNzc5fQ%3D%3D--8a34d432bd73868779898a451aeb2b06a8ccc335)

**Result**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/aa.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9hYS5wbmciLCJleHBpcmVzIjoxNDA5MjEwODE2fQ%3D%3D--66b78b06206f3737025cb272a18ea54a398b7d37)

#3. 2D and 3D	
##3.1. MAKE POINT
 The point made by make_float2 or make_float3 has *struct* data type. A float2 variable has two members x and y, a float3 variable has three members x, y and z.  A float3 variable can be used to make RGB color pixel. If variable name is p, then *p.x*, *p.y* and *p.z* correspond to red, green and blue.

- **make_nDataType(nDpoint)**
- make_float2(x, y)
- make_float3(x, y, z)
- make_uchar2(x, y)
- make_uchar3(x, y, z)


**Example**
```python
def gradation1(x, y):
    ret = make_float3(x/2, y/2, 100)
    return ret
def gradation2(x, y):
    ret = make_float3(y/2, 0, x/2)
    return ret
def gradation3(x, y):
    ret = make_float3(255, x/2, y/2)
    return ret

def main():
    result1 = gradation1(x, y, z).range(x=0:510, y=0:510)
    result2 = gradation2(x, y, z).range(x=0:510, y=0:510)
    result3 = gradation3(x, y, z).range(x=0:510, y=0:510)
    save_image(result1,'gradation1.png')
    save_image(result2,'gradation2.png')
    save_image(result3,'gradation3.png')
```

**Reult**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/color1.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9jb2xvcjEucG5nIiwiZXhwaXJlcyI6MTQwOTIxMDg1MH0%3D--533fd16b69cd1211f792a6522fddfc7a411c064d)

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/color2.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9jb2xvcjIucG5nIiwiZXhwaXJlcyI6MTQwOTIxMDg3N30%3D--5990238a563aec7710791d10e814df10463fc7be)

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/color3.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9jb2xvcjMucG5nIiwiZXhwaXJlcyI6MTQwOTIxMDg5MX0%3D--49739138d0433bab50b91ba8024503f3c5d94436)

##3.2. SAMPLERS	
 Since input data is defined on a rectilinear grid, Vivaldi provides various memory object samplers for hardware-accelerated interpolation.

 **point_query_nd()** is used to access the data value nearest to the given location (or sampled at the integer coordinates) from an n-dimensional memory object.

  **linear/cubic_query_nD()** implements fast cubic interpolation based on the technique by Sigg et al. it is often used in volume rendering when sampling at arbitrary locations for discrete data defined on an n-dimensional rectilinear grid.

### 2D QUERY	
- point_query_2d(T* image, float2 p)
- linear_query_2d(T* image, float2 p)

**Example**
```python
def pq2(image, x, y):
    return point_query_2d(image, x, y)

def lq2(image, x, y):
	return linear_query_2d(image, x+0.5, y+0.5)

def main():
    image = load_data_2d(DATA_PATH+'image.jpg')
    result = pq2(image,x,y).range(image).dtype(image,uchar)
	result2 = lq2(image,x,y).range(image).dtype(image,uchar)
    save_image(result,'pq2.png')
	save_image(result2,'lq2.png')
```

**Result**

Point query 2d
![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/pq2.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9wcTIucG5nIiwiZXhwaXJlcyI6MTQwOTIxODY4MH0%3D--ae8aebad5fa5660096e002711ab724bdaf807204)

Linear query 2d
![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/lq2.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9scTIucG5nIiwiZXhwaXJlcyI6MTQwOTIxODY0Mn0%3D--8518e08ec1f3cd75e5edfc70fe5152dbb6ecdb72)

### 3D QUERY	
- point_query_3d(T* volume, float3 p)

**Example**
```python
def pq3(image, x, y, z):
    return point_query_3d(image, x, y, z)

def main():
    import numpy
    image = numpy.ones((5, 5, 5), dtype=numpy.float32)
	# image is  x + y + z, 5x5x5, 3D matrix
    for i in range(5):
        for j in range(5):
            for k in range(5):
                image[i][j][k] = i + j + k
	
	result = lq3(image, x, y, z).range(x=0:5,y=0:5,z=0:5)
								.dtype(image, float)

	print “result[3][3][3] is 3 + 3 + 3 :” + str(result[3][3][3])
```
**Result**
```
result[3][3][3] is 3 + 3 + 3 : 9.0
```

- linear_query_3d(T* volume, float3 p)

**Example**
```python
def lq3(data, x, y, z):
    return linear_query_3d(data, x+0.5, y+0.5, z+0.5)

def main():
    import numpy
    data = numpy.ones((8,8,8),dtype=numpy.float32)

    for x in range(8):
        for y in range(8):
            for z in range(8):
                data[x][y][z]=x+y+z

    #input data is 8x8x8x, output result is 6x6x6
    result = lq3(data, x, y, z).range(x=1:7,y=1:7,z=1:7).dtype(data, float)

    correct = numpy.zeros((6,6,6), dtype=numpy.float32)
	#linear query is 8 point query
    for i in range(6):
        for j in range(6):
            for k in range(6):
                x = i+1
                y = j+1
                z = k+1
                a = data[x][y][z]
                b = data[x][y+1][z]
                c = data[x+1][y][z]
                d = data[x+1][y+1][z]
                e = data[x][y][z+1]
                f = data[x][y+1][z+1]
                g = data[x+1][y][z+1]
                h = data[x+1][y+1][z+1]
                correct[i][j][k] = (a+b+c+d+e+f+g+h)/8
  
    if (correct == result).all():
        print "Ok"
    else:
        print "Fail"
```
**Result**
```
OK
```

- cubic_query_3d(T* volume, float3 p)
**Example**
```python
def cq3(data, x, y, z):
    return cubic_query_3d(data, x, y, z)

def lq3(data, x, y, z):
    return linear_query_3d(data, x, y, z)

def main():
    # Qubic query 3d is 8 linear query 3d
    import numpy
    data = numpy.ones((8,8,8),dtype=numpy.float32)

    for x in range(8):
        for y in range(8):
            for z in range(8):
                data[x][y][z]=x+y+z

    #input data is 8x8x8, output result is 6x6x6
    result = cq3(data, x, y, z).range(x=1:7,y=1:7,z=1:7).dtype(data, float)

    correct = numpy.zeros((6,6,6), dtype=numpy.float32)
    #cubic query is 8 linear query
    for i in range(6):
        for j in range(6):
            for k in range(6):
                x = i+1
                y = j+1
                z = k+1
                a = lq3(data, x, y, z)
                b = lq3(data, x, y+0.5, z)
                c = lq3(data, x+0.5, y, z)
                d = lq3(data, x+0.5, y+0.5, z)
                e = lq3(data, x, y, z+0.5)
                f = lq3(data, x, y+0.5, z+0.5)
                g = lq3(data, x+0.5, y, z+0.5)
                h = lq3(data, x+0.5, y+0.5, z+0.5)
                correct[i][j][k] = (a+b+c+d+e+f+g+h)/8
    print correct

    if (correct == result).all():
        print "Ok"
    else:       
        print "Fail"
```
**Result**
```
?
```

##3.3. QUERY WITH BORDER	
Functions are used to perform various linear or non-linear filtering operations on 2D images, so neighborhood of pixel location in the image is considered. 
The computed response is stored in the destination image at the same location (x,y). It means that the output image will be of the same size as the input image.

This functions compute and return the coordinate of a donor pixel corresponding to the specified extrapolated pixel when using the specified extrapolation border mode.
- point_query_2d(image, float2 p, BORDER_REFLECT)
- point_query_2d(image, float2 p, BORDER_REFLECT_101)
- point_query_2d(image, float2 p, BORDER_REPLICATE)
- point_query_2d(image, float2 p, BORDER_WRAP)

 Various border types, image boundaries are denoted with '|'
 * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
 * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
 * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
 * BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
 * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'

**Example**
```python
def original(image, x, y):
    return point_query_2d(image, x, y)
def border_reflect(image, x, y):
    return point_query_2d(image, x, y, BORDER_REFLECT)
def border_reflect_101(image, x, y):
    return point_query_2d(image, x, y, BORDER_REFLECT_101)
def border_replicate(image, x, y):
    return point_query_2d(image, x, y, BORDER_REPLICATE)
def border_wrap(image, x, y):
    return point_query_2d(image, x, y, BORDER_WRAP)

def main():
	#size of image is 1280x853
    image = load_data_2d(DATA_PATH+'image.jpg')
	#ORIGINAL
    result = original(image, x, y).range(x=-600:1880,y=-400:1253).dtype(image, uchar)
    save_image(result,'original.png')

    # BORDER_REFLECT
    result = border_reflect(image, x, y).range(x=-600:1880,y=-400:1253).dtype(image,uchar)
    save_image(result,'reflect.png')

    # BORDER_REFLECT_101
    result = border_reflect_101(image, x, y).range(x=-600:1880,y=-400:1253).dtype(image, uchar)
    save_image(result,'reflect_101.png')

    # BORDER_REPLICATE
    result = border_replicate(image, x, y).range(x=-600:1880,y=-400:1253).dtype(image,uchar)
    save_image(result,'replicate.png')

    # BORDER_WRAP
    result = border_wrap(image, x, y).range(x=-600:1880,y=-400:1253).dtype(image,uchar)
    save_image(result,'wrap.png')
```
**Original**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/origin.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9vcmlnaW4ucG5nIiwiZXhwaXJlcyI6MTQwOTIxOTY2N30%3D--c53600d21d0f206888e2cd754bd08efe0c54acc5)

**Reflect**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/reflect.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9yZWZsZWN0LnBuZyIsImV4cGlyZXMiOjE0MDkyMTE5NzZ9--5df4472a4c56dd45e56b6f1eb769a326e32bbd30)

**Reflect_101**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/reflect_101.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9yZWZsZWN0XzEwMS5wbmciLCJleHBpcmVzIjoxNDA5MjEyMDEwfQ%3D%3D--d7716a49249535f1d29341cd4efe3c5133da907b)

**Replicate**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/replicate.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9yZXBsaWNhdGUucG5nIiwiZXhwaXJlcyI6MTQwOTIxMjA0MX0%3D--095cf43b3650783731f2f54e2ce155a55699eeb3)

**Wrap**
![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/wrap.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS93cmFwLnBuZyIsImV4cGlyZXMiOjE0MDkyMTIwNjd9--927b91d583681f66d72f7a1ceccca4036a94a87c)

#4. ITERATORS	
 Vivaldi provides several iterator abstractions to access neighborhood values near each data location. ***User-defined iterators*** are used for iteration on *line-*, *plane-*, and *cube-* shaped neighborhood regions.

 Vivaldi’s built in ***viewer-defined iterators*** are used for iteration on a ray generated from the center of each screen pixel depending on the current projection mode, such as *orthogonal-* and *perspective-* projections.

##4.1.	LINE ITERATOR	
Line iterator creates a *user-defined iterator* that starts from point **start** and moves a distance **step** along the line segment **(start, end)**.
- line_iter(float3 start, float3 end, float step)

**Example**
```python
def line_iter_test(x, y):
    start = make_float3(x, y, 0)
    end = make_float3(x, y, 10)
    iter = line_iter(start, end, 1)
    val = 0
    for elem in iter:
        val += 1
    return val
def main():
    result = line_iter_test(x, y).range(x=0:5, y=0:5)
    print result
```
**Result**
```
[[10 10 10 10 10]
 [10 10 10 10 10]
 [10 10 10 10 10]
 [10 10 10 10 10]
 [10 10 10 10 10]]
```

##4.2.	PLANE ITERATOR	
Plane iterator creates a *user-defined iterator* that square shaped which center is point **center** and radius is value of **radius**. If radius is 2, the size of square is 5x5. 
- plane_iter(float2 center, float radius)

**Example**
```python
def plane_iter_test(data, x, y):
    iter = plane_iter(x, y, 1)
    sum = 0.0
	for elem in iter:
    	sum += point_query_2d(data, elem)
    return sum

def main():
    import numpy
	data = numpy.ones((5,5), dtype=numpy.float32)
    result = plane_iter_test(data, x, y).range(x=0:5,y=0:5)
    print result
```
**Result**
```
[[ 4.  6.  6.  6.  4.]
 [ 6.  9.  9.  9.  6.]
 [ 6.  9.  9.  9.  6.]
 [ 6.  9.  9.  9.  6.]
 [ 4.  6.  6.  6.  4.]]
```

**Example**
```python
#mean filter
def mean(image, x, y):
    radius = 5
    val = make_float3(0)
    sum = make_float3(0)
    iter = plane_iter(x, y, radius)
    for point in iter:
        val = point_query_2d(image, point)
        sum += val

    num = (2*radius+1)*(2*radius+1)
    sum = sum / num
    return sum

def main():
	#size of image is 1280x720
    image = load_data_2d(DATA_PATH+'image.jpg')
    result = mean(image,x,y).range(image).dtype(image, uchar)
    save_image(result,'mean.png')
```

**Original**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/dd.jpg?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9kZC5qcGciLCJleHBpcmVzIjoxNDA5MjEyMTUwfQ%3D%3D--8e31d9a49dc7bfbe0241a0b78a9508139a29f3f3)

**Result**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/mean.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9tZWFuLnBuZyIsImV4cGlyZXMiOjE0MDkyMTIxMjJ9--c7398516a9af041ed4d7ce00cd01c82cc2373ad8)

##4.3.	CUBE ITERATOR	
Cube iterator creates a *user-defined iterator* that cube shaped which center is point **center** and radius is value of **radius**. If radius is 2, the size of cube is 5x5x5.
- cube_iter(float3 center, float radius)

**Example**
```python
def cube_iter_test(data, x,y,z):
    point = make_float3(x,y,z)
    radius = 1
    iter = cube_iter(point, radius)
    sum = 0.0
    for elem in iter:
        val = point_query_3d(data, elem)
        sum += val
    return sum

def main():
    import numpy as np
    data = numpy.ones((5,5,5),dtype=numpy.float32)
    result = cube_iter_test(data, x,y,z).range(data)
										.dtype(data,float)
	print "result[0][0][0] is " + str(result[0][0][0])
    print "result[1][1][1] is " + str(result[1][1][1])
    print "result[2][2][2] is " + str(result[2][2][2])
```
**Result**
```
result[0][0][0] is 8.0
result[1][1][1] is 27.0
result[2][2][2] is 27.0
```
**Example**
```python
	radius = 2
```
**Result**
```
result[0][0][0] is 27.0
result[1][1][1] is 64.0
result[2][2][2] is 125.0
```

##4.4.	ORTHOGONAL ITERATOR	
Orthogonal iterator returns a line iterator for a ray originated from each pixel location x, y and parallel to the viewing direction. The default eye location is on the positive z-axis looking at the origin, and the volume is centered at origin. 

  The location and orientation of volume can also be changed using model transformation function **Rotate()** and **Translate()**. When using rotate function, the axis of rotation is *direction*. 
- orthogonal_iter(T* volume, float2 orgin, float step)
- Rotate(float angle, float3 direction)
- Translate(float3 direction)

**Example**
```python
def mip(volume, x, y):
    step = 1.0
    iter = orthogonal_iter(volume, x, y, step)
    max = 0.0
    for elem in iter:
        val = linear_query_3d(volume, elem)
        if max < val:
            max = val
    return max

def main():
    volume = load_data_3d(DATA_PATH+'/lobster.dat')
	result = mip(volume, x, y).range(x=-128:128, y=-128:128)
							  .dtype(volume, uchar)
    save_image(result,'orthogonal.png')
```

**Result**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/orthogonal.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9vcnRob2dvbmFsLnBuZyIsImV4cGlyZXMiOjE0MDkyMTIyOTh9--e4e7ec3bbcca6252cb5364adc4dbc6c75fa7b8d7)

**Example**
```python
	Rotate(90, 0, 0, 1)
	Translate(-128, -128, 0)
```

**Result**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/rotate.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9yb3RhdGUucG5nIiwiZXhwaXJlcyI6MTQwOTIxOTcxOX0%3D--9d52fc5801c707619babac96ccdb6db0447a2f49)

**Example**
```python
   Translate(-128, -128, 0)
```

**Result**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/translate.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS90cmFuc2xhdGUucG5nIiwiZXhwaXJlcyI6MTQwOTIxMjQ2N30%3D--3529b390acabe283d16ba619e8631b06a83e0bc5)

##4.5.	PERSPECTIVE ITERATOR	
Perspective iterator creates a line iterator for a line segment defined by the intersection of a viewing ray and the 3D volume cube to be rendered.

*near* is distance between viewport and eyes and size of viewport is fixed.
- perspective_iter(T* volume, float2 orgin, float step, float near)

**Example**
```python
def mip(volume, x, y, near):
    step = 1.0
    iter = perspective_iter(volume, x, y, step, near)
    max = 0.0
    for elem in iter:
        val = linear_query_3d(volume, elem)
        if max < val:
            max = val
    return max

def main():
    volume = load_data_3d(DATA_PATH+'/lobster.dat')
    Translate(-128, -128, 0)

    result0 = mip(volume,x,y,1).range(x=-256:256,y=-256:256).dtype(volume, uchar)
    result1 = mip(volume,x,y,10).range(x=-256:256,y=-256:256).dtype(volume, uchar)
    result2 = mip(volume,x,y,50).range(x=-256:256,y=-256:256).dtype(volume, uchar)
    result3 = mip(volume,x,y,100).range(x=-256:256,y=-256:256).dtype(volume, uchar)

    save_image(result0,'result0.png')
    save_image(result1,'result1.png')
    save_image(result2,'result2.png')
    save_image(result3,'result3.png')
```

**Result**

near = 1

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/1.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS8xLnBuZyIsImV4cGlyZXMiOjE0MDkyOTE0ODF9--23f9cb7a92b7475b0114f38500807da3b80c22b5)

near = 10

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/10.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS8xMC5wbmciLCJleHBpcmVzIjoxNDA5MjkxNTExfQ%3D%3D--8e88f18b873eb5439c446b9a12f8f723274b21f7)

near = 50

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/50.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS81MC5wbmciLCJleHBpcmVzIjoxNDA5MjkxNTI4fQ%3D%3D--e0615d9c575e42c3b5f85fff886b6af3468ac637)

near = 100

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/100.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS8xMDAucG5nIiwiZXhwaXJlcyI6MTQwOTI5MTUyNH0%3D--6e3d9ac62dc12cc4d007884b0a3da264393fe970)


#5.	MODIFIER	
Vivaldi memory objects serve as input and output buffers for worker functions. Although arbitrarily many inputs can be used for a function there is only one output object per function execution. The function execution can be configured using various ***execution modifiers*** using the following syntax.

**output = function(input, parameters).execution modifiers**

The execution modifiers describe how the output values for each input element are generated in a *data-parallel* fashion similar to GL shaders or CUDA kernels.

##5.1.	RANGE	
Specifies the size of the output memory object.
- output = function(input, parameters).range(x=0:n, y=0:m)
- output = function(input, parameters).range(input)

**Example**
```python
def function(image, x, y):
    return  point_query_2d(image, x, y)

def main():
    image = load_data_2d(DATA_PATH+'image.jpg')
	#size of image is 1280x720
    result = function(image, x, y).range(x=300:900,y=200:500)
                                  .dtype(image, uchar)
    save_image(result, 'result.png')
```

**Result**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/cut.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9jdXQucG5nIiwiZXhwaXJlcyI6MTQwOTIxNzk2M30%3D--1a7f69560dd5a94a93ccfb4c66791bc22400ec96)

**Example**
```python
def function(image, x, y):
    return  point_query_2d(image, x, y)

def main():
    image = load_data_2d(DATA_PATH+'image.jpg')
    result = function(image, x, y).range(image)
								  .dtype(image, uchar)
    save_image(result, 'result.png')
```

**Result**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/dd.jpg?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9kZC5qcGciLCJleHBpcmVzIjoxNDA5MjE4MDA3fQ%3D%3D--6c796f875c9aa08804016a8a46ed5df076f748b3)

##5.2.	DTYPE
Specify the type of input data. *DTYPE* is not always nesessary because Vivaldi automatically checks data type.
- output = function(input, parameters).dtype(input, data_type)

**Example**
```python
def function(data, x, y)
    val = point_query_2d(data, x, y)
    return val

def main():
    import numpy as np
    data = np.ones((5,5), dtype=numpy.float32)
    result = function(data, x, y).range(x=0:5, y=0:5)
                                 .dtype(data, float)
    print result
```
**Result**
```
[[ 1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.]]
```

##5.3.	EXECID	
Specifies the execution ID of the unit where the function is run.
- gpu_list = get_GPU_list(2)
- output = function(input, parameters).execid(gpu_list)

**Example**
```python
def get_id(x, y):
    return DEVICE_NUMBER

def main():
    gpu_list = get_GPU_list(2)
    result = get_id(x, y).range(x=0:6,y=0:6)
                         .split(result, x=2,y=2)
                         .execid(gpu_list)
    print result
```
**Result**
```
[[ 0.  0.  0.  1.  1.  1.]
 [ 0.  0.  0.  1.  1.  1.]
 [ 0.  0.  0.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.  1.]]
```
**Example**
```python
	gpu_list = get_GPU_list(4)
	result = function(x, y).range(x=0:5,y=0:5)
                           .split(result, x=2,y=2)
                           .execid(gpu_list)
    print result
```
**Result**
```
[[ 0.  0.  1.  1.  1.]
 [ 0.  0.  1.  1.  1.]
 [ 2.  2.  3.  3.  3.]
 [ 2.  2.  3.  3.  3.]
 [ 2.  2.  3.  3.  3.]]
```

##5.4.	SPLIT	
Specifies parallel execution by splitting input or/and output memory objects.
###Input split
 The input data is split into partitions, and each partition is used to generate an output of same size. Because multiple output data are generated for the same output region, there must be an additional merging step to consolidate the data using the **merge** execution modifier.

 This model works well when the input data size is very large and the output data size is relatively small. *Sort-last parallel volume rendering* is a good example.
- output = function(input).split(input, x=2, y=2).merge(function2)

**Example**
```python
def function(data, x, y):
return point_query_2d(data, x, y)

def multi(front, back, x, y):
    f = point_query_2d(front, x, y)
    b = point_query_2d(back, x, y)
    return f + b

def main():
    import numpy as np
    data = np.ones((6,6), dtype=numpy.float32)
	#input split
    result = function(data, x, y).range(x=0:6,y=0:6)
                                 .dtype(data, float)
                                 .split(data, x=2)
                                 .merge(multi, 'front-to-back')
    print result
```
**Result**
```
[[ 1.  1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.  1.]]
```


###Output split
 The input data is duplicated for each task and each task generates a subset of the output data. Since the output from each task does not overlap with other outputs, there is no merge function required. 

 An example of this model is *sort-first parallel volume rendering* where each task is holding the entire input data and renders only a sub-region of the screen. 
- output = function(input).split(output, x=2, y=2)

**Example**
```python
def function(data, x, y):
    return DEVICE_NUMBER*10

def main():
    import numpy
    data = numpy.ones((8,8), dtype=numpy.float32)
    #output split
    result = function(data, x, y).range(x=0:8, y=0:8)
								 .dtype(data, float)
 								 .split(result, x=2, y=2)
	print result
```
**Result**
```
[[ 0  0  0  0 10 10 10 10]
 [ 0  0  0  0 10 10 10 10]
 [ 0  0  0  0 10 10 10 10]
 [ 0  0  0  0 10 10 10 10]
 [20 20 20 20 30 30 30 30]
 [20 20 20 20 30 30 30 30]
 [20 20 20 20 30 30 30 30]
 [20 20 20 20 30 30 30 30]]
```

###In-and-out split
 Both input and output data are split into same number of identical partitions. This model applies well to *data-parallel problems* and *isosurface extraction*.

 Since each task only needs to store a small subset of input data, this model can handle very large input if many parallel execution units are available.
- output = function(input).split(input, x=2, y=2).split(output, x=2, y=2)

**Example**
```python
#compare each excution time
#no split data vs. in-and-out split data
def function(data, x, y):
    return DEVICE_NUMBER*100

def main():
    size = 1000
    import numpy
    data = numpy.ones((size,size), dtype=numpy.float32)
    #no split
    result1 = function(data, x, y).range(x=0:size, y=0:size)
								  .dtype(data, float)
	#in-and-out split (using 4 gpu)
	result2 = function(data, x, y).range(x=0:size, y=0:size)
								  .dtype(data, float)
								  .split(data, x=2, y=2)
								  .split(result2, x=2, y=2)
	print “result1 is ” + str(result1)
	print “result2 is ” + str(result2)
```
**Result**
```
reult1 is
[[0 0 0 ..., 0 0 0]
 [0 0 0 ..., 0 0 0]
 [0 0 0 ..., 0 0 0]
 ...,
 [0 0 0 ..., 0 0 0]
 [0 0 0 ..., 0 0 0]
 [0 0 0 ..., 0 0 0]]
TIME 12.1511569023
result2 is
[[  0   0   0 ..., 100 100 100]
 [  0   0   0 ..., 100 100 100]
 [  0   0   0 ..., 100 100 100]
 ...,
 [200 200 200 ..., 300 300 300]
 [200 200 200 ..., 300 300 300]
 [200 200 200 ..., 300 300 300]]
TIME 0.595387935638
```

###Multiple-in-and-out split
 Multiple input data are split into different partitions, but the combination of all the input partitions matches the split partitions of the output data. Each task stores two pieces of half the input data and a quarter of the output data. An example of this parallel model is the *multiplication of two matrices*.
- output = function(input1, input2).split(input1, y=2).split(input2, x=2).split(output, x=2, y=2)

**Example**
```python
#multiplication of two 2D matrix
def matrix_multi(data1, data2, x, y):
    outputval = 0.0
    row = 0.0
    col = 0.0
    cnt = 0
    for elem in range(6):
        row = point_query_2d(data1, cnt, y)
        col = point_query_2d(data2, x, cnt)
        outputval += row * col
        cnt += 1
    return outputval

def main():
	size = 6
    import numpy as np
    data1 = np.ones((size,size), dtype=numpy.float32)
    data2 = np.ones((size,size), dtype=numpy.float32)
    data1[2][3]=5
    data2[1][2]=7
    output = matrix_multi(data1, data2, x, y).range(x=0:size, y=0:size)
											 .dtype(data1, float)
											 .dtype(data2, float)
											 .split(data1,y=2)
											 .split(data2,x=2)
											 .split(output, x=2,y=2)
	print data1
	print "multiply"
	print data2 
	print "is"
	print output
```
**Result**
```
[[  1.   1.   1.   1.   1.   1.]
 [  1.   1.   1.   1.   1.   1.]
 [  1.   1.   1.   5.   1.   1.]
 [  1.   1.   1.   1.   1.   1.]
 [  1.   1.   1.   1.   1.   1.]
 [  1.   1.   1.   1.   1.   1.]]
multiply
[[  1.   1.   1.   1.   1.   1.]
 [  1.   1.   7.   1.   1.   1.]
 [  1.   1.   1.   1.   1.   1.]
 [  1.   1.   1.   1.   1.   1.]
 [  1.   1.   1.   1.   1.   1.]
 [  1.   1.   1.   1.   1.   1.]]
is
[[  6.   6.  12.   6.   6.   6.]
 [  6.   6.  12.   6.   6.   6.]
 [ 10.  10.  16.  10.  10.  10.]
 [  6.   6.  12.   6.   6.   6.]
 [  6.   6.  12.   6.   6.   6.]
 [  6.   6.  12.   6.   6.   6.]]
```
 
##5.5.	MERGE	
Specifies a user-defined merging function.
- output = function(input, parameters).merge(func, 'front-to-back')

**Example**
```python
def function(data, x, y):
    left = point_query_2d(data, x-1, y)
    right = point_query_2d(data, x+1, y)
    val = left and right
    return val
def multi(front, back, x, y):
    f = point_query_2d(front, x, y)
    b = point_query_2d(back, x, y)
    return f + b

def main():
    import numpy as np
    data = np.ones((8,8), dtype=numpy.float32)

    result = function(data, x, y).range(x=0:8,y=0:8)
                                 .dtype(data, float)
                                 .split(data, x=2)
                                 .merge(multi, 'front-to-back')
    print result
```
**Result**
```
[[ 0.  1.  1.  0.  0.  1.  1.  0.]
 [ 0.  1.  1.  0.  0.  1.  1.  0.]
 [ 0.  1.  1.  0.  0.  1.  1.  0.]
 [ 0.  1.  1.  0.  0.  1.  1.  0.]
 [ 0.  1.  1.  0.  0.  1.  1.  0.]
 [ 0.  1.  1.  0.  0.  1.  1.  0.]
 [ 0.  1.  1.  0.  0.  1.  1.  0.]
 [ 0.  1.  1.  0.  0.  1.  1.  0.]]
```

##5.6.	HALO COMMUNICATOR	
 Decomposition can be easily implemented using the **split** execution modifier, but there might be additional communication required across neighborhood regions depending on the computation type. Extra regions storing neighbor values are called *halo*. Vivaldi also provides the **halo** execution modifier for automatic and implicit communication between halos.
 
 An *input halo* is the extra region around the input data usually defined by the operator size. An *output halo* is the extra region around the output region usually used for running multiple iterations without halo communication.
- output = function(input, parameters).split(input, x=2).halo(input, size).halo(output, size)

**Example**
```python
def function(data, x, y):
    left = point_query_2d(data, x-1, y)
    right = point_query_2d(data, x+1, y)
    val = left and right
    return val

def main():
    import numpy as np
    data = np.ones((8,8), dtype=numpy.float32)
    result = function(data, x, y).range(x=0:8, y=0:8)
                                 .dtype(data, float)
                                 .split(data, x=2)
                                 .split(result, x=2)
    print result
```
**Result**
```
[[ 0.  1.  1.  0.  0.  1.  1.  0.]
 [ 0.  1.  1.  0.  0.  1.  1.  0.]
 [ 0.  1.  1.  0.  0.  1.  1.  0.]
 [ 0.  1.  1.  0.  0.  1.  1.  0.]
 [ 0.  1.  1.  0.  0.  1.  1.  0.]
 [ 0.  1.  1.  0.  0.  1.  1.  0.]
 [ 0.  1.  1.  0.  0.  1.  1.  0.]
 [ 0.  1.  1.  0.  0.  1.  1.  0.]]
```
**Example**
```python
	result = function(data, x, y).range(x=0:8, y=0:8)
                	             .dtype(data, float)
            	                 .split(data, x=2)
        	                     .split(result, x=2)
    	                         .halo(data, 1)
	print result
```
**Result**
```
[[ 0.  1.  1.  1.  1.  1.  1.  0.]
 [ 0.  1.  1.  1.  1.  1.  1.  0.]
 [ 0.  1.  1.  1.  1.  1.  1.  0.]
 [ 0.  1.  1.  1.  1.  1.  1.  0.]
 [ 0.  1.  1.  1.  1.  1.  1.  0.]
 [ 0.  1.  1.  1.  1.  1.  1.  0.]
 [ 0.  1.  1.  1.  1.  1.  1.  0.]
 [ 0.  1.  1.  1.  1.  1.  1.  0.]]
```

#6.	DIFFERENTIAL OPERATORS	
Vivaldi provides first and second order differential operators frequently used in numerical computing.

##6.1.	GRADIENT	
First order differential operator to compute partial derivatives in the n-dimensional Vivaldi memory object using linear/cubic interpolation that returns an n-tuple vector.
- linear_gradient_2d(T* image, float2 p)
- linear_gradient_3d(T* volume, float3 p)
- cubic_gradient_3d(T* vllume, float3 p)

**Example**
```python
def edge_detection(image, x, y):
    diff = linear_gradient_2d(image, x, y)
    if length(diff) > 20:
        return 255
    else:
        return 0

def main():
    image = load_data_2d(DATA_PATH+'image.jpg')
    result = edge_detection(image, x, y).range(image).dtype(image,uchar)
    save_image(result, 'lg2.png')
```

**Original**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/cc.jpg?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9jYy5qcGciLCJleHBpcmVzIjoxNDA5MjE4MjI0fQ%3D%3D--9d9fd7448546a2ebd068f6361171597c7d5cd16d)

**Result**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/lg2.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9sZzIucG5nIiwiZXhwaXJlcyI6MTQwOTIxODE4OX0%3D--1391ce531f962a32026f5e25fb7f7ce870e29d09)

**Example**
```python
def surface_detection(volume, x, y):
    location = make_float3(x, y, 100)
    diff = linear_gradient_3d(volume, location)
    if length(diff) > 1:
        return 1
    else:
        return 0

def main():
    volume = load_data_3d(DATA_PATH+'lobster.dat')
    result = surface_detection(volume, x, y).range(x=0:256,y=0:256).dtype(volume,uchar)
    save_image(result, 'lg3.png')
```

**Result**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/lg3_1.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9sZzNfMS5wbmciLCJleHBpcmVzIjoxNDA5MjE4Mjk2fQ%3D%3D--c34dbe8f52a0fda37d73bfd60109e4eb997ec069)

**Example**
```python
	if length(diff) > 10:
		return 1
	else:
		return 0
```

**Result**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/lg3_10.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9sZzNfMTAucG5nIiwiZXhwaXJlcyI6MTQwOTIxODMxM30%3D--f6389138792965d60fac83b90b14bde216060362)

##6.2.	LAPLACIAN	
Second order differential operator to compute the divergence of the gradient (Laplace operator). Calculate laplacian using near 4 points.
- laplacian(T* image, float2 p, VIVALDI_DATA_RANGE* sdr)

**Example**
```python
def heatflow(image, x, y):
    a = laplacian(image, x, y)
    ret = point_query_2d(image, x, y)
    dt = 0.000025
    for i in range(10000):
        ret = ret + dt*a
    return ret

def main():
    image = load_data_2d(DATA_PATH+'image.jpg')
    #Heatflow iteration
    for i in range(10):
        image = heatflow(image, x, y).range(image).dtype(image,uchar)
        #change input data type float to uchar
        image = numpy.array(image, dtype=numpy.uint8)
	
    save_image(image,'heatflow.png')
```

**Result**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/heatflow.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9oZWF0Zmxvdy5wbmciLCJleHBpcmVzIjoxNDA5MjE4MzQ0fQ%3D%3D--c088f579a1bd987ffdecd3f7ea32d27c3f47a6b3)

#7.	SHADING MODELS	
Vivaldi provides two built-in shading models to easily achieve different shading effects for surface rendering.

##7.1.	PHONG	
Calculate phong shading color using light position, normal vector and etc.
- phong(float3 Light_position, float3 pos, float3 N, float3 omega, float3 kd, float3 ks, float n, float3 amb)

**Example**
```python
def phong(volume, x, y):
    step = 0.4
    line_iter = orthogonal_iter(volume, x, y, step)
    D = make_float3(0)
    D = line_iter.direction()
    T = 50
    ambient = make_float3(0.05)
    lightpos1 = make_float3( -800, -800, 0)
    lightpos2 = make_float3( 300, -100, 0)

    for elem in line_iter:
        temp = cubic_query_3d(volume, elem)-T
        if temp > 0:
            P = elem
            Q = P
            P = P - step * D

            while length(P - Q) >  0.0001:
                M = make_float3(0)
                M = (P + Q) / 2
                f = linear_query_3d(volume, M) - T
                if f < 0:
                    P = M
                else:
                    Q = M

            color1 = make_float3(225.0/255,73.0/255,15.0/255)
            color2 = make_float3(1,1,1)
            tmp_pq = make_float3(0)
            tmp_pq = (P+Q)/2
            N = make_float3(0)
            N = normalize(cubic_gradient_3d(volume,tmp_pq))
            N = -N

            L1 = normalize(lightpos1 - P)
            L2 = normalize(lightpos2 - P)

            # accumulate
            result1 = phong(L1, N, -D, color1, make_float3(1), 20, ambient)*255
            result2 = phong(L2, N, -D, color2, make_float3(1), 20, ambient)*255

            return RGB(result1 + result2)

    return RGB(make_float3(255))

def main():
    volume = load_data_3d('lobster.dat')
	Rotate(150, 1, 0, 0)
	result = phong(volume, x, y).range(x=0:256,y=-256:0).dtype(volume,uchar)
    save_image(result,'result.png')
```

**Result**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/phong.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9waG9uZy5wbmciLCJleHBpcmVzIjoxNDA5MzAxNzc3fQ%3D%3D--089397d288b70fe8aa3181198f816f478b158ca8)

##7.2.	DIFFUSE	
Calculate diffuse only using light position and normal vector.
- diffuse(float3 Light_position, float3 N, float3 kd)

**Example**
```python
def diffuse(volume, x, y):
	step = 0.4                      
	line_iter = orthogonal_iter(volume, x, y, step)
	D = make_float3(0)
	D = line_iter.direction()
	T = 50

	ambient = make_float3(0.05)
	lightpos1 = make_float3( -800, -800, 0)
	lightpos2 = make_float3( 300, -100, 0)
	
	for elem in line_iter:
		temp = point_query_3d(volume, elem)-T
		if temp > 0:
			P = elem
			Q = P
			P = P - step * D

			while length(P - Q) >  0.0001:
				M = make_float3(0)
				M = (P + Q) / 2
				f = point_query_3d(volume, M) - T
				if f < 0:
					P = M
				else:
					Q = M

			color1 = make_float3(225.0/255,15.0/255,15.0/255)
			color2 = make_float3(1,1,1)
			N = make_float3(0)
			N = normalize(cubic_gradient_3d(volume,(P+Q)/2))
			N = -N
			
			L1 = normalize(lightpos1 - P)
			L2 = normalize(lightpos2 - P)

			# accumulate
			result1 = diffuse(L1, N, color1)*255
			result2 = diffuse(L2, N, color2)*255
			return RGB(result1 + result2)

	return RGB(make_float3(255))
							  
def main():                                                                         
	volume = load_data_3d(DATA_PATH+'/lobster.dat')

	Rotate(150, 1, 0, 0)
	result = diffuse(volume, x, y).range(x=0:256,y=-256:0).dtype(volume, uchar)
	save_image(result)
```

**Result**

![](https://raw.githubusercontent.com/hvcl/Vivaldi/master/image/diffuse.png?token=8486747__eyJzY29wZSI6IlJhd0Jsb2I6aHZjbC9WaXZhbGRpL21hc3Rlci9pbWFnZS9kaWZmdXNlLnBuZyIsImV4cGlyZXMiOjE0MDkzMDE3MTl9--0513fb58a8f54e5aa30ac7814822c908c800abde)

