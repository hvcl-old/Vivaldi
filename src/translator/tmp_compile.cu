#include "/home/hschoi/Vivaldi/src/py-src/helper_math.h"
#define GPU inline __device__
#define uchar unsigned char

#define BORDER_REPLICATE 1
#define BORDER_REFLECT 2
#define BORDER_REFLECT_101 3
#define BORDER_WRAP 4

#include<stdio.h>

__device__ int DEVICE_NUMBER;
__device__ float modelview[4][4];
__device__ float inv_modelview[4][4];

__device__ float TF_bandwidth;
__device__ int front_back;


// clamp
GPU float clamp(float f, float a, float b);
GPU float2 clamp(float2 v, float a, float b);
GPU float3 clamp(float3 v, float a, float b);
GPU float3 clamp(float3 v, float3 a, float3 b);
GPU float4 clamp(float4 v, float a, float b);
GPU float4 clamp(float4 v, float4 a, float4 b);

// uchar2
//////////////////////////////////////////////////////////////////////
GPU uchar2 make_uchar2(float2 s);
// uchar3
//////////////////////////////////////////////////////////////////////
GPU uchar3 make_uchar3(float3 s);
// uchar4
//////////////////////////////////////////////////////////////////////
GPU uchar4 make_uchar4(float4 s);

// short2
//////////////////////////////////////////////////////////////////////
GPU short2 make_short2(float2 s);
// short3
//////////////////////////////////////////////////////////////////////
GPU short3 make_short3(float3 s);
// short4
//////////////////////////////////////////////////////////////////////
GPU short4 make_short4(float4 s);

// ushort2
//////////////////////////////////////////////////////////////////////
GPU ushort2 make_ushort2(float2 s);
// ushort3
//////////////////////////////////////////////////////////////////////
GPU ushort3 make_ushort3(float3 s);
// ushort4
//////////////////////////////////////////////////////////////////////
GPU ushort4 make_ushort4(float4 s);


// float2
/////////////////////////////////////////////////////////////////////

GPU float2 make_float2(uchar2 s);
GPU float2 make_float2(uchar3 s);
GPU float2 make_float2(uchar4 s);

GPU float2 make_float2(int s);
GPU float2 make_float2(float s);
GPU float2 make_float2(int2 a);

// float3
/////////////////////////////////////////////////////////////////////
GPU float3 make_float3(uchar2 s);
GPU float3 make_float3(uchar3 s);
GPU float3 make_float3(uchar4 s);

GPU float3 make_float3(float s);
GPU float3 make_float3(float2 a);
GPU float3 make_float3(float2 a, float s);
GPU float3 make_float3(float3 a);
GPU float3 make_float3(float4 a);
GPU float3 make_float3(int3 a);

// float4
/////////////////////////////////////////////////////////////////////
GPU float4 make_float4(uchar2 s);
GPU float4 make_float4(uchar3 s);
GPU float4 make_float4(uchar4 s);

GPU float4 make_float4(int s);
GPU float4 make_float4(float s);
GPU float4 make_float4(float2 a);
GPU float4 make_float4(float3 a);
GPU float4 make_float4(float4 a);
GPU float4 make_float4(int3 a);

// rgba class
class RGBA{
public:
    unsigned char r, g, b, a;
	GPU RGBA(float3 rgb, float a_in)
	{
		r = clamp(rgb.x, 0.0f, 255.0f);
		g = clamp(rgb.y, 0.0f, 255.0f);
		b = clamp(rgb.z, 0.0f, 255.0f);
		a = clamp(a_in, 0.0f, 255.0f);
	}
	GPU RGBA(float4 rgba)
	{
		r = clamp(rgba.x, 0.0f, 255.0f);
		g = clamp(rgba.y, 0.0f, 255.0f);
		b = clamp(rgba.z, 0.0f, 255.0f);
		a = clamp(rgba.w, 0.0f, 255.0f);
	}
	GPU RGBA(float r_in, float g_in, float b_in, float a_in)
	{
		r = clamp(r_in, 0.0f, 255.0f);
		g = clamp(g_in, 0.0f, 255.0f);
		b = clamp(b_in, 0.0f, 255.0f);
		a = clamp(a_in, 0.0f, 255.0f);
	}
	GPU RGBA(float c)
	{
		r = clamp(c, 0.0f, 255.0f);
		g = clamp(c, 0.0f, 255.0f);
		b = clamp(c, 0.0f, 255.0f);
		a = clamp(255.0f, 0.0f, 255.0f);
	}

	GPU RGBA()
    {
        r = g = b = 0;
        a = 1;
    }

};

// rgb class
class RGB{
public:
    unsigned char r, g, b;
	GPU RGB(float3 rgb)
	{
		r = clamp(rgb.x, 0.0f, 255.0f);
		g = clamp(rgb.y, 0.0f, 255.0f);
		b = clamp(rgb.z, 0.0f, 255.0f);
	}
	GPU RGB(float4 rgb)
	{
		r = clamp(rgb.x, 0.0f, 255.0f);
		g = clamp(rgb.y, 0.0f, 255.0f);
		b = clamp(rgb.z, 0.0f, 255.0f);
	}

	GPU RGB(float r_in, float g_in, float b_in)
	{
		r = clamp(r_in, 0.0f, 255.0f);
		g = clamp(g_in, 0.0f, 255.0f);
		b = clamp(b_in, 0.0f, 255.0f);
	}
	GPU RGB(float a)
    {
        r = g = b = a;
    }
	GPU RGB(int a)
    {
        r = g = b = a;
    }
	GPU RGB()
    {
        r = g = b = 0;
    }

};

class VIVALDI_DATA_RANGE{
public:
	int4 start, end; // world coordinate
	int4 full_buffer_start, full_buffer_end; // world coordinate
	int halo;
	int buffer_halo;
	
};



// data type converters 
////////////////////////////////////////////////////////////////////////////////
GPU float convert(unsigned char a){
	return float(a);
}
GPU float convert(short a){
	return float(a);
}
GPU float convert(ushort a){
	return float(a);
}
GPU float convert(int a){
	return float(a);
}
GPU float convert(float a){
	return float(a);
}
GPU float convert(double a){
	return float(a);
}

GPU float2 convert(uchar2 a){
    return make_float2(a);
}
GPU float2 convert(short2 a){
    return make_float2(a.x,a.y);
}
GPU float2 convert(ushort2 a){
    return make_float2(a.x,a.y);
}
GPU float2 convert(int2 a){
    return make_float2(a.x,a.y);
}
GPU float2 convert(float2 a){
    return make_float2(a.x,a.y);
}

GPU float3 convert(RGB a){
    return make_float3(a.r,a.g,a.b);
}
GPU float3 convert(uchar3 a){
    return make_float3(a);
}
GPU float3 convert(short3 a){
    return make_float3(a.x,a.y,a.z);
}
GPU float3 convert(ushort3 a){
    return make_float3(a.x,a.y,a.z);
}
GPU float3 convert(int3 a){
    return make_float3(a.x,a.y,a.z);
}
GPU float3 convert(float3 a){
    return make_float3(a.x,a.y,a.z);
}

GPU float4 convert(RGBA a){
    return make_float4(a.r,a.g,a.b,a.a);
}
GPU float4 convert(uchar4 a){
    return make_float4(a);
}
GPU float4 convert(short4 a){
	return make_float4(a.x,a.y,a.z,a.w);
}
GPU float4 convert(ushort4 a){
	return make_float4(a.x,a.y,a.z,a.w);
}
GPU float4 convert(int4 a){
    return make_float4(a.x,a.y,a.z,a.w);
}
GPU float4 convert(float4 a){
    return make_float4(a.x,a.y,a.z,a.w);
}

// data_type init
//////////////////////////////////////////////////////////////
GPU float initial(unsigned char a){
	return float(0);
}
GPU float initial(short a){
	return float(0);
}
GPU float initial(ushort a){
	return float(0);
}
GPU float initial(int a){
	return float(0);
}
GPU float initial(float a){
	return float(0);
}
GPU float initial(double a){
	return float(0);
}

GPU float2 initial(uchar2 a){
	return make_float2(0);
}
GPU float2 initial(short2 a){
	return make_float2(0);
}
GPU float2 initial(ushort2 a){
	return make_float2(0);
}
GPU float2 initial(int2 a){
	return make_float2(0);
}
GPU float2 initial(float2 a){
	return make_float2(0);
}

GPU float3 initial(RGB a){
    return make_float3(0);
}
GPU float3 initial(uchar3 a){
    return make_float3(0);
}
GPU float3 initial(short3 a){
    return make_float3(0);
}
GPU float3 initial(ushort3 a){
    return make_float3(0);
}
GPU float3 initial(int3 a){
    return make_float3(0);
}
GPU float3 initial(float3 a){
    return make_float3(0);
}

GPU float4 initial(RGBA a){
	return make_float4(0);
}
GPU float4 initial(uchar4 a){
	return make_float4(0);
}
GPU float4 initial(short4 a){
    return make_float4(0);
}
GPU float4 initial(ushort4 a){
    return make_float4(0);
}
GPU float4 initial(int4 a){
    return make_float4(0);
}
GPU float4 initial(float4 a){
    return make_float4(0);
}

GPU float initial2(unsigned char a){
	return float(1);
}
GPU float initial2(short a){
	return float(1);
}
GPU float initial2(ushort a){
	return float(1);
}
GPU float initial2(int a){
	return float(1);
}
GPU float initial2(float a){
	return float(1);
}
GPU float initial2(double a){
	return float(1);
}

GPU float2 initial2(uchar2 a){
	return make_float2(1);
}
GPU float2 initial2(short2 a){
	return make_float2(1);
}
GPU float2 initial2(ushort2 a){
	return make_float2(1);
}
GPU float2 initial2(int2 a){
	return make_float2(1);
}
GPU float2 initial2(float2 a){
	return make_float2(1);
}

GPU float3 initial2(RGB a){
    return make_float3(1);
}
GPU float3 initial2(uchar3 a){
    return make_float3(1);
}
GPU float3 initial2(short3 a){
    return make_float3(1);
}
GPU float3 initial2(ushort3 a){
    return make_float3(1);
}
GPU float3 initial2(int3 a){
    return make_float3(1);
}
GPU float3 initial2(float3 a){
    return make_float3(1);
}

GPU float4 initial2(RGBA a){
	return make_float4(1);
}
GPU float4 initial2(uchar4 a){
	return make_float4(1);
}
GPU float4 initial2(short4 a){
    return make_float4(1);
}
GPU float4 initial2(ushort4 a){
    return make_float4(1);
}
GPU float4 initial2(int4 a){
    return make_float4(1);
}
GPU float4 initial2(float4 a){
    return make_float4(1);
}


// float functions
////////////////////////////////////////////////////////////////////////////////

GPU float length(float a){
	return sqrt(a*a);
}

//f = value, a = min, b = max
GPU float step(float edge, float x){
    return x < edge ? 0 : 1;
}

GPU float rect(float edge0, float edge1, float x){
    return edge0 <= x && x <= edge1 ? 1 : 0;
}

// uchar
GPU uchar make_uchar(float s){
	return (unsigned char)s;
}

GPU uchar make_uchar(float2 s){
	return (unsigned char)s.x;
}

GPU uchar make_uchar(float3 s){
	return (unsigned char)s.x;
}

GPU uchar make_uchar(float4 s){
	return (unsigned char)s.x;
}

// uchar2
//////////////////////////////////////////////////////////////////////
GPU uchar2 make_uchar2(float2 s){
	return make_uchar2(s.x,s.y);
}
// uchar3
//////////////////////////////////////////////////////////////////////
GPU uchar3 make_uchar3(float3 s){
	return make_uchar3(s.x,s.y,s.z);
}
// uchar4
//////////////////////////////////////////////////////////////////////
GPU uchar4 make_uchar4(float4 s){
	return make_uchar4(s.x,s.y,s.z,s.w);
}


// short2
//////////////////////////////////////////////////////////////////////
GPU short2 make_short2(float2 s){
	return make_short2(s.x,s.y);
}
// short3
//////////////////////////////////////////////////////////////////////
GPU short3 make_short3(float3 s){
	return make_short3(s.x,s.y,s.z);
}
// short4
//////////////////////////////////////////////////////////////////////
GPU short4 make_short4(float4 s){
	return make_short4(s.x,s.y,s.z,s.w);
}
// ushort2
//////////////////////////////////////////////////////////////////////
GPU ushort2 make_ushort2(float2 s){
	return make_ushort2(s.x,s.y);
}
// ushort3
//////////////////////////////////////////////////////////////////////
GPU ushort3 make_ushort3(float3 s){
	return make_ushort3(s.x,s.y,s.z);
}
// ushort4
//////////////////////////////////////////////////////////////////////
GPU ushort4 make_ushort4(float4 s){
	return make_ushort4(s.x,s.y,s.z,s.w);
}

// int2 functions
GPU int2 make_int2(int4 s){
	return make_int2(int(s.x), int(s.y));
}

// int3 functions
GPU int3 make_int3(int4 s){
	return make_int3(int(s.x), int(s.y), int(s.z));
}

// float2 functions
////////////////////////////////////////////////////////////////////////////////

GPU float2 make_float2(uchar2 s){
	return make_float2(float(s.x),float(s.y));
}

GPU float2 make_float2(uchar3 s){
	return make_float2(float(s.x),float(s.y));
}

GPU float2 make_float2(uchar4 s){
	return make_float2(float(s.x),float(s.y));
}

GPU float2 make_float2(int s){
    return make_float2(s, s);
}

// negate
GPU float2 operator-(float2 a){
    return make_float2(-a.x, -a.y);
}

// floor
GPU float2 floor(const float2 v){
    return make_float2(floor(v.x), floor(v.y));
}

// reflect
GPU float2 reflect(float2 i, float2 n){
	return i - 2.0f * n * dot(n,i);
}

// float3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
GPU float3 make_float3(uchar2 s){
	return make_float3(float(s.x),float(s.y), float(0));
}

GPU float3 make_float3(uchar3 s){
	return make_float3(float(s.x),float(s.y),float(s.z));
}

GPU float3 make_float3(uchar4 s){
	return make_float3(float(s.x),float(s.y),float(s.z));
}

// floor
GPU float3 floor(const float3 v){
    return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

// float4 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
GPU float4 make_float4(uchar2 s){
	return make_float4(float(s.x),float(s.y), float(0), float(0));
}

GPU float4 make_float4(float2 s){
	return make_float4(float(s.x),float(s.y), float(0), float(0));
}

GPU float4 make_float4(uchar3 s){
	return make_float4(float(s.x),float(s.y),float(s.z),float(0));
}

GPU float4 make_float4(uchar4 s){
	return make_float4(float(s.x),float(s.y),float(s.z),float(s.w));
}

GPU float4 make_float4(float4 s){
	return make_float4(float(s.x),float(s.y),float(s.z),float(s.w));
}

GPU float4 make_float4(int s){
    return make_float4(s, s, s, s);
}

GPU float4 make_float4(float a, float s){
    return make_float4(a, a, a, s);
}

// negate
GPU float4 operator-(float4 a){
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

// floor
GPU float4 floor(const float4 v){
    return make_float4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}


// Frame
////////////////////////////////////////////////////////////////////////////////

class Frame{
public:
    float3 x, y, z, origin;
    
    GPU void setDefault(float3 position)
    {
        origin = position;
        x = make_float3(1, 0, 0);
        y = make_float3(0, 1, 0);
        z = make_float3(0, 0, 1);
    }
    
    GPU void lookAt(float3 position, float3 target, float3 up)
    {
        origin = position;
        z = normalize(position - target);
        x = normalize(cross(up, z));
        y = normalize(cross(z, x));
    }
    
    GPU float3 getVectorToWorld(float3 v)
    {
        return v.x * x + v.y * y + v.z * z;
    }
    
    GPU float3 getPointToWorld(float3 p)
    {
        return p.x * x + p.y * y + p.z * z + origin;
    }
};

// transfer2
////////////////////////////////////////////////////////////////////////////////

GPU float transfer2(
    float x0, float f0,
    float x1, float f1,
    float x)
{
    if (x < x0) return 0;
    if (x < x1) return lerp(f0, f1, (x - x0) / (x1 - x0));
    return 0;
}

GPU float2 transfer2(
    float x0, float2 f0,
    float x1, float2 f1,
    float x)
{
    if (x < x0) return make_float2(0);
    if (x < x1) return lerp(f0, f1, (x - x0) / (x1 - x0));
    return make_float2(0);
}

GPU float3 transfer2(
    float x0, float3 f0,
    float x1, float3 f1,
    float x)
{
    if (x < x0) return make_float3(0);
    if (x < x1) return lerp(f0, f1, (x - x0) / (x1 - x0));
    return make_float3(0);
}

GPU float4 transfer2(
    float x0, float4 f0,
    float x1, float4 f1,
    float x)
{
    if (x < x0) return make_float4(0);
    if (x < x1) return lerp(f0, f1, (x - x0) / (x1 - x0));
    return make_float4(0);
}


// transfer3
////////////////////////////////////////////////////////////////////////////////

GPU float transfer3(
    float x0, float f0,
    float x1, float f1,
    float x2, float f2,
    float x)
{
    if (x < x0) return 0;
    if (x < x1) return lerp(f0, f1, (x - x0) / (x1 - x0));
    if (x < x2) return lerp(f1, f2, (x - x1) / (x2 - x0));
    return 0;
}

GPU float2 transfer3(
    float x0, float2 f0,
    float x1, float2 f1,
    float x2, float2 f2,
    float x)
{
    if (x < x0) return make_float2(0);
    if (x < x1) return lerp(f0, f1, (x - x0) / (x1 - x0));
    if (x < x2) return lerp(f1, f2, (x - x1) / (x2 - x0));
    return make_float2(0);
}

GPU float3 transfer3(
    float x0, float3 f0,
    float x1, float3 f1,
    float x2, float3 f2,
    float x)
{
    if (x < x0) return make_float3(0);
    if (x < x1) return lerp(f0, f1, (x - x0) / (x1 - x0));
    if (x < x2) return lerp(f1, f2, (x - x1) / (x2 - x0));
    return make_float3(0);
}

GPU float4 transfer3(
    float x0, float4 f0,
    float x1, float4 f1,
    float x2, float4 f2,
    float x)
{
    if (x < x0) return make_float4(0);
    if (x < x1) return lerp(f0, f1, (x - x0) / (x1 - x0));
    if (x < x2) return lerp(f1, f2, (x - x1) / (x2 - x0));
    return make_float4(0);
}

// helper textures for cubic interpolation and random numbers
////////////////////////////////////////////////////////////////////////////////

texture<float4, 2, cudaReadModeElementType> hgTexture;
texture<float4, 2, cudaReadModeElementType> dhgTexture;
texture<int, 2, cudaReadModeElementType> randomTexture;

GPU float3 hg(float a){
    // float a2 = a * a;
    // float a3 = a2 * a;
    // float w0 = (-a3 + 3*a2 - 3*a + 1) / 6;
    // float w1 = (3*a3 - 6*a2 + 4) / 6;
    // float w2 = (-3*a3 + 3*a2 + 3*a + 1) / 6;
    // float w3 = a3 / 6;
    // float g = w2 + w3;
    // float h0 = (1.0f + a) - w1 / (w0 + w1);
    // float h1 = (1.0f - a) + w3 / (w2 + w3);
    // return make_float3(h0, h1, g);
    return make_float3(tex2D(hgTexture, a, 0));
}
GPU float3 dhg(float a){
    return make_float3(tex2D(dhgTexture, a, 0));
}

//iterators
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class line_iter{
public:
	float3 S,E,P,step;
	float len;
	GPU line_iter(){
	}
	GPU line_iter(float3 from, float3 to, float d){
		S = from;
		E = to;
		step = normalize(to-from)*d;
		len = length(to-from);
		P = S;
		if( S.x == E.x &&
			S.y == E.y &&
			S.z == E.z )step = make_float3(0,0,1);
	}
	GPU float3 begin(){
		return S;
	}
	GPU bool hasNext(){
		float3 T = P + step;
		if( length(T-S) > len)return false;
		return true;
	}
	GPU float3 next(){
		P += step;
		return P;
	}
	GPU float3 direction(){
		return normalize(step);
	}
};
// class make_laplacian_iter{}
class make_plane_iter{
public:
	float2 S;
	float d;
	int max_step, step;
	int width;
	float x,y;
	GPU make_plane_iter(float2 point, float size){
		S = point;
		d = size;
		width = 1+2*size;
		max_step = width*width;
	}
	GPU make_plane_iter(int x, int y, float size){
		S = make_float2(x,y);
		d = size;
		width = 1+2*size;
		max_step = width*width;
	}
	GPU make_plane_iter(float x, float y, float size){
		S = make_float2(x,y);
		d = size;
		width = 1+2*size;
		max_step = width*width;
	}

	GPU float2 begin(){
		step = 0;
		x = 0;
		y = 0;
		return S + make_float2(-d,-d);
	}
	GPU bool hasNext(){
		if(max_step == step)return false;
		return true;
	}
	GPU float2 next(){
		step++;
		x++;
		if( x == width){ x=0; y++;}
		float2 P = S + make_float2( x - d, y - d);
		return P;
	}
};
class make_cube_iter{
public:
	float3 S;
	int d;
	int width;
	int max_step, step;
	float x,y,z;
	GPU make_cube_iter(float3 point, float size){
		S = point;
		d = size;
		width = 1+2*size;
		max_step = (width)*(width)*(width);
	}
	GPU make_cube_iter(int x,int y,int z, float radius){
		S = make_float3(x,y,z);
		d = radius;
		width = 1+2*radius;
		max_step = (width)*(width)*(width);
	
	}
	GPU make_cube_iter(float x, float y, float z, float size){
		S = make_float3(x,y,z);
		d = size;
		width = 1+2*size;
		max_step = (width)*(width)*(width);
	
	}
	GPU float3 begin(){
		step = 0;
		x = 0;
		y = 0;
		z = 0;
		return S + make_float3(-d, -d, -d);
	}
	GPU bool hasNext(){
		if(max_step == step)return false;
		return true;
	}
	GPU float3 next(){
		step++;
		x++;
		if( x == width){ x=0; y++;}
		if( y == width){ y=0; z++;}

		float3 P = S + make_float3( x - d, y - d, z - d);
		return P;
	}
};

// data query functions
//////////////////////////////////////////////////////////////////////////////
#define INF            __int_as_float(0x7f800000)

GPU int2 float2_to_int2(float2 a){
	return make_int2(int(a.x), int(a.y));
}
GPU int3 float3_to_int3(float3 a){
    return make_int3(int(a.x), int(a.y), int(a.z));
}


// BORDER handling functions
GPU int border_replicate(int x, int pivot){
	// aaaaaa|abcdefgh|hhhhhhh
	return pivot;
}
GPU int border_reflect(int x, int pivot){
	// fedcba|abcdefgh|hgfedcb
	int a;
	a = 0;
	if(x < pivot) a = -1;
	if(x > pivot) a = 1;
		
	return 2*pivot-x + a;
}
GPU int border_reflect_101(int x, int pivot){
	// gfedcb|abcdefgh|gfedcba
	return 2*pivot-x;
}
GPU int border_wrap(int x, int pivot, int w){
	// cdefgh|abcdefgh|abcdefg
	if(x > pivot)return x - w;
	if(x < pivot)return x + w;
	return pivot;
}
GPU int border_constant(){
// iiiiii|abcdefgh|iiiiiii  with some specified 'i'
	return 0;
}

GPU int border_switch(int x, int pivot, int w, int border){

	switch(border){
		case BORDER_REPLICATE: // replicate
			return border_replicate(x, pivot);
		case BORDER_REFLECT: // reflect
			return border_reflect(x, pivot);
		case BORDER_REFLECT_101: // reflect_101
			return border_reflect_101(x, pivot);
		case BORDER_WRAP: // border_wrap
			return border_wrap(x, pivot, w);
		default: // border_constant
			return border_constant();
	}
}

GPU int check_in_border(int p, int start, int end){
	if(p > end-1)return 1; // right
	else if(p < start)return -1; // left
	return 0; // middle
}

GPU int border_handling(int p, int start, int end, int border){
	int flag = -1;
	while(flag != 0){
		flag = check_in_border(p, start, end);
		
		if(flag == 1){
			p = border_switch(p, end-1, end - start, border);
		}else if(flag == -1){
			p = border_switch(p, start, end - start, border);
		}
	}
	return p;
}


// 2D data query functions
////////////////////////////////////////////////////////////////////////////////
template<typename R,typename T> GPU R point_query_2d(T* data, float2 p, int border, VIVALDI_DATA_RANGE* sdr){

	int2 sub_buffer_start = make_int2(sdr->start);
	int2 sub_buffer_end = make_int2(sdr->end);

	int2 data_start = make_int2(sdr->full_buffer_start);
	int2 data_end = make_int2(sdr->full_buffer_end);

	int x = p.x;
	int y = p.y;
	
	int X = sub_buffer_end.x - sub_buffer_start.x;
	R rt;
	
	// input coordinate is world coordinate
	
	// border handling
	int flag1, flag2; // -1, 0, 1
	flag1 = check_in_border(x, data_start.x, data_end.x);
	flag2 = check_in_border(y, data_start.y, data_end.y);
	
	bool flag; // border 
	flag = (1 <= border && border <= 4);
	
	if(flag){
		x = border_handling(x, data_start.x, data_end.x, border);
		y = border_handling(y, data_start.y, data_end.y, border);		
		
		x -= sub_buffer_start.x;
		y -= sub_buffer_start.y;
		rt = convert(data[y*X + x]);
	}else{
		if(flag1 == 0 && flag2 == 0){
			x -= sub_buffer_start.x;
			y -= sub_buffer_start.y;
			rt = convert(data[y*X + x]);
		}else{
			rt = initial2(data[0])*0;
		}
	}
	return rt;
}
template<typename R,typename T> GPU R point_query_2d(T* image, float2 p, VIVALDI_DATA_RANGE* sdr){
	return point_query_2d<R>(image, p, 0, sdr);
}
template<typename R,typename T> GPU R point_query_2d(T* image, float x, float y, int border, VIVALDI_DATA_RANGE* sdr){
	return point_query_2d<R>(image, make_float2(x,y), border, sdr);
}
template<typename R,typename T> GPU R point_query_2d(T* image, float x, float y, VIVALDI_DATA_RANGE* sdr){
	return point_query_2d<R>(image, make_float2(x,y), 0, sdr);
}

template<typename R,typename T> GPU R linear_query_2d(T* image, float2 p, int border, VIVALDI_DATA_RANGE* sdr){

	int2 sub_buffer_start = make_int2(sdr->start);
	int2 sub_buffer_end = make_int2(sdr->end);

	int2 data_start = make_int2(sdr->full_buffer_start);
	int2 data_end = make_int2(sdr->full_buffer_end);

	//range check
	float x = p.x;
	float y = p.y;

	int fx = floor(x);
	int fy = floor(y);
	int cx = ceil(x);
	int cy = ceil(y);

	float dx = x - fx;
	float dy = y - fy;

	R iv = initial(image[0])*0;
	
	R q00 = point_query_2d<R>(image, fx, fy, border, sdr);
	R q01 = point_query_2d<R>(image, cx, fy, border, sdr);
	R q10 = point_query_2d<R>(image, fx, cy, border, sdr);
	R q11 = point_query_2d<R>(image, cx, cy, border, sdr);
	
	// lerp along x
	R q0 = lerp(q00, q01, dx);
	R q1 = lerp(q10, q11, dx);

	// lerp along y
	R q = lerp(q0, q1, dy);
	return q;
}
template<typename R, typename T> GPU R linear_query_2d(T* image, float2 p, VIVALDI_DATA_RANGE* sdr){
	return linear_query_2d<R>(image, p, 0, sdr);
}
template<typename R, typename T> GPU R linear_query_2d(T* image, float x, float y, int border, VIVALDI_DATA_RANGE* sdr){
	return linear_query_2d<R>(image, make_float2(x,y), border, sdr);
}
template<typename R, typename T> GPU R linear_query_2d(T* image, float x, float y, VIVALDI_DATA_RANGE* sdr){
	return linear_query_2d<R>(image, make_float2(x,y), 0, sdr);
}

template<typename R, typename T> GPU float2 linear_gradient_2d(T* image, float2 p, VIVALDI_DATA_RANGE* sdr){

	int2 start = make_int2(sdr->start);
	int2 end = make_int2(sdr->end);

	int2 data_start = make_int2(sdr->full_buffer_start);
	int2 data_end = make_int2(sdr->full_buffer_end);

	int halo = sdr->halo;

	float x = p.x - start.x;
    float y = p.y - start.y;
    int X = end.x - start.x;
    int Y = end.y - start.y;

    float2 rbf = make_float2(0);
  
	if( x < halo)return rbf;
    if( y < halo)return rbf;
    if( x >= X-halo)return rbf;
    if( y >= Y-halo)return rbf;
 
	float delta = 1.0f;

	R xf, xb;
	xf = linear_query_2d<R>(image, make_float2(p.x + delta, p.y), sdr);
	xb = linear_query_2d<R>(image, make_float2(p.x - delta, p.y), sdr);
	float dx = length(xf-xb);

	R yf, yb;
	yf = linear_query_2d<R>(image, make_float2(p.x, p.y + delta), sdr);
	yb = linear_query_2d<R>(image, make_float2(p.x, p.y - delta), sdr);
	float dy = length(yf-yb);

	return make_float2(dx,dy)/(2*delta);
}
template<typename R, typename T> GPU float2 linear_gradient_2d(T* image, float x, float y, VIVALDI_DATA_RANGE* sdr){
	return linear_gradient_2d<R>(image, make_float2(x,y), sdr);
}
template<typename R, typename T> GPU float2 linear_gradient_2d(T* image, int x, int y, VIVALDI_DATA_RANGE* sdr){
	return linear_gradient_2d<R>(image, make_float2(x,y), sdr);
}

// 3D data query functions
////////////////////////////////////////////////////////////////////////////////
template<typename R, typename T> GPU R point_query_3d(T* image, float3 p, int border, VIVALDI_DATA_RANGE* sdr){
	int3 sub_buffer_start = make_int3(sdr->start);
	int3 sub_buffer_end = make_int3(sdr->end);

	int3 data_start = make_int3(sdr->full_buffer_start);
	int3 data_end = make_int3(sdr->full_buffer_end);
	
	int x = p.x;
	int y = p.y;
	int z = p.z;
	
	int X = sub_buffer_end.x - sub_buffer_start.x;
	int Y = sub_buffer_end.y - sub_buffer_start.y;
	R rt;


	// buffer_coordinate input
	
	int buffer_halo = sdr->buffer_halo;
	// border handling
	//x -= sub_buffer_start.x;
	//y -= sub_buffer_start.y;
	//z -= sub_buffer_start.z;
	
	int flag1, flag2, flag3; // -1, 0, 1
	flag1 = check_in_border(x, data_start.x+buffer_halo, data_end.x+buffer_halo);
	flag2 = check_in_border(y, data_start.y+buffer_halo, data_end.y+buffer_halo);
	flag3 = check_in_border(z, data_start.z+buffer_halo, data_end.z+buffer_halo);
	
	bool flag; // border 
	flag = (1 <= border && border <= 4);
	
	if(flag){
		x = border_handling(x, data_start.x+buffer_halo, data_end.x+buffer_halo, border);
		y = border_handling(y, data_start.y+buffer_halo, data_end.y+buffer_halo, border);
		z = border_handling(z, data_start.z+buffer_halo, data_end.z+buffer_halo, border);
		
		rt = convert(image[z*Y*X+ y*X + x]);
	}else{
		if(flag1 == 0 && flag2 == 0 && flag3 == 0){
			rt = convert(image[z*Y*X + y*X + x]);
		}else{
			rt = initial2(image[0])*(0);
		}
	}
	
	// world coordinate input
	
	// border handling
	//x -= sub_buffer_start.x;
	//y -= sub_buffer_start.y;
	//z -= sub_buffer_start.z;
	/*
	int flag1, flag2, flag3; // -1, 0, 1
	flag1 = check_in_border(x, data_start.x, data_end.x);
	flag2 = check_in_border(y, data_start.y, data_end.y);
	flag3 = check_in_border(z, data_start.z, data_end.z);
	
	flag3 = 1;
	
	bool flag; // border 
	flag = (1 <= border && border <= 4);
	
	if(flag){
		x = border_handling(x, data_start.x, data_end.x, border);
		y = border_handling(y, data_start.y, data_end.y, border);
		z = border_handling(z, data_start.z, data_end.z, border);
		
		rt = convert(image[z*Y*X+ y*X + x]);
	}else{
		if(flag1 == 0 && flag2 == 0 && flag3 == 0){
			x -= sub_buffer_start.x;
			y -= sub_buffer_start.y;
			z -= sub_buffer_start.z;
			rt = convert(image[z*Y*X + y*X + x]);
		}else{
			rt = initial2(image[0])*x;
		}
	}
	*/
	return rt;
}
template<typename R, typename T> GPU R point_query_3d(T* image, float3 p, VIVALDI_DATA_RANGE* sdr){
	return point_query_3d<R>(image, p, 0, sdr);
}
template<typename R, typename T> GPU R point_query_3d(T* image, float x, float y, float z, int border, VIVALDI_DATA_RANGE* sdr){
	return point_query_3d<R>(image, make_float3(x,y,z), border, sdr);
}
template<typename R, typename T> GPU R point_query_3d(T* image, float x, float y, float z, VIVALDI_DATA_RANGE* sdr){
	return point_query_3d<R>(image, make_float3(x,y,z), 0, sdr);
}

GPU float d_lerp(float a, float b, float t){
    return (b - a) * t;
}

template<typename R, typename T> GPU R linear_query_3d(T* volume, float3 p, int border, VIVALDI_DATA_RANGE* sdr){
    //tri linear interpolation
    float x,y,z;
    x = p.x;
    y = p.y;
    z = p.z;
	
    int fx = floor(x);
    int fy = floor(y);
    int fz = floor(z);
    int cx = ceil(x);
    int cy = ceil(y);
    int cz = ceil(z);
	
	R q000 = point_query_3d<R>(volume, fx, fy, fz, border, sdr);
	R q001 = point_query_3d<R>(volume, fx, fy, cz, border, sdr);
	R q010 = point_query_3d<R>(volume, fx, cy, fz, border, sdr);
	R q011 = point_query_3d<R>(volume, fx, cy, cz, border, sdr);
	R q100 = point_query_3d<R>(volume, cx, fy, fz, border, sdr);
	R q101 = point_query_3d<R>(volume, cx, fy, cz, border, sdr);
	R q110 = point_query_3d<R>(volume, cx, cy, fz, border, sdr);
	R q111 = point_query_3d<R>(volume, cx, cy, cz, border, sdr);

    float dx = x - fx;
    float dy = y - fy;
    float dz = z - fz;

    // lerp along x
    R q00 = lerp(q000, q001, dx);
    R q01 = lerp(q010, q011, dx);
    R q10 = lerp(q100, q101, dx);
    R q11 = lerp(q110, q111, dx);

    // lerp along y
    R q0 = lerp(q00, q01, dy);
    R q1 = lerp(q10, q11, dy);

    // lerp along z
    R q = lerp(q0, q1, dz);
    return q;
}
template<typename R, typename T> GPU R linear_query_3d(T* volume, float3 p, VIVALDI_DATA_RANGE* sdr){
	return linear_query_3d<R>(volume, p, 0, sdr);
}
template<typename R, typename T> GPU R linear_query_3d(T* volume, float x, float y, float z, int border, VIVALDI_DATA_RANGE* sdr){
	return linear_query_3d<R>(volume, make_float3(x,y,z), border, sdr);
}
template<typename R, typename T> GPU R linear_query_3d(T* volume, float x, float y, float z, VIVALDI_DATA_RANGE* sdr){
	return linear_query_3d<R>(volume, make_float3(x,y,z), 0, sdr);
}

template<typename R, typename T> GPU float3 linear_gradient_3d(T* volume, float3 p, VIVALDI_DATA_RANGE* sdr){
	int3 start = make_int3(sdr->start);
	int3 end = make_int3(sdr->end);

	int3 data_start = make_int3(sdr->full_buffer_start);
	int3 data_end = make_int3(sdr->full_buffer_end);

//	int halo = sdr->halo;

    int X = end.x - start.x;
    int Y = end.y - start.y;
	int Z = end.z - start.z;

    float3 rbf = make_float3(0);

	float x1 = p.x;
	float y1 = p.y;
	float z1 = p.z;

	float x,y,z;
	x = inv_modelview[0][0] * x1 + inv_modelview[0][1] * y1 + inv_modelview[0][2] * z1 + inv_modelview[0][3];
	y = inv_modelview[1][0] * x1 + inv_modelview[1][1] * y1 + inv_modelview[1][2] * z1 + inv_modelview[1][3];
	z = inv_modelview[2][0] * x1 + inv_modelview[2][1] * y1 + inv_modelview[2][2] * z1 + inv_modelview[2][3];

	x = x - start.x;
	y = y - start.y;
	z = z - start.z;
/*
	if( x < halo)return rbf;
    if( y < halo)return rbf;
	if( z < halo)return rbf;
    if( x >= X-halo)return rbf;
    if( y >= Y-halo)return rbf;
	if( z >= Z-halo)return rbf;
*/
	if( x < 0)return rbf;
    if( y < 0)return rbf;
	if( z < 0)return rbf;
    if( x >= X)return rbf;
    if( y >= Y)return rbf;
	if( z >= Z)return rbf;

	x = p.x - start.x;
    y = p.y - start.y;
	z = p.z - start.z;
 
	float delta = 1.0f;
	R dx = linear_query_3d<R>(volume, make_float3(p.x + delta, p.y, p.z), sdr) - linear_query_3d<R>(volume, make_float3(p.x - delta, p.y, p.z), sdr);
	R dy = linear_query_3d<R>(volume, make_float3(p.x, p.y + delta, p.z), sdr) - linear_query_3d<R>(volume, make_float3(p.x, p.y - delta, p.z), sdr);
	R dz = linear_query_3d<R>(volume, make_float3(p.x, p.y, p.z + delta), sdr) - linear_query_3d<R>(volume, make_float3(p.x, p.y, p.z - delta), sdr);

	float dxl = length(dx);
	float dyl = length(dy);
	float dzl = length(dz);
	return make_float3(dxl, dyl, dzl) / (2 * delta);
}
template<typename R, typename T> GPU float3 linear_gradient_3d(T* volume, int x, int y, int z, VIVALDI_DATA_RANGE* sdr){
	return linear_gradient_3d<R>(volume, make_float3(x,y,z), sdr);
}
template<typename R, typename T> GPU float3 linear_gradient_3d(T* volume, float x, float y, float z, VIVALDI_DATA_RANGE* sdr){
	return linear_gradient_3d<R>(volume, make_float3(x,y,z), sdr);
}

template<typename R,typename T> GPU T cubic_query_3d(T* volume, float3 p, VIVALDI_DATA_RANGE* sdr){


	int3 start = make_int3(sdr->start);
	int3 end = make_int3(sdr->end);

	int3 data_start = make_int3(sdr->full_buffer_start);
	int3 data_end = make_int3(sdr->full_buffer_end);




	float3 alpha = 255 * (p - floor(p - 0.5f) - 0.5f);

	float3 hgx = hg(alpha.x);
	float3 hgy = hg(alpha.y);
	float3 hgz = hg(alpha.z);

	// 8 linear queries
	R q000 = linear_query_3d<R>(volume, p.x - hgx.x, p.y - hgy.x, p.z - hgz.x, sdr);
	R q001 = linear_query_3d<R>(volume, p.x - hgx.x, p.y - hgy.x, p.z + hgz.y, sdr);
	R q010 = linear_query_3d<R>(volume, p.x - hgx.x, p.y + hgy.y, p.z - hgz.x, sdr);
	R q011 = linear_query_3d<R>(volume, p.x - hgx.x, p.y + hgy.y, p.z + hgz.y, sdr);
	R q100 = linear_query_3d<R>(volume, p.x + hgx.y, p.y - hgy.x, p.z - hgz.x, sdr);
	R q101 = linear_query_3d<R>(volume, p.x + hgx.y, p.y - hgy.x, p.z + hgz.y, sdr);
	R q110 = linear_query_3d<R>(volume, p.x + hgx.y, p.y + hgy.y, p.z - hgz.x, sdr);
	R q111 = linear_query_3d<R>(volume, p.x + hgx.y, p.y + hgy.y, p.z + hgz.y, sdr);

	// lerp along z
	R q00 = lerp(q000, q001, hgz.z);
	R q01 = lerp(q010, q011, hgz.z);
	R q10 = lerp(q100, q101, hgz.z);
	R q11 = lerp(q110, q111, hgz.z);

	// lerp along y
	R q0 = lerp(q00, q01, hgy.z);
	R q1 = lerp(q10, q11, hgy.z);

	// lerp along x
	R q = lerp(q0, q1, hgx.z);
	return q;
}
template<typename R,typename T> GPU float3 cubic_gradient_3d(T* data, float3 p, VIVALDI_DATA_RANGE* sdr){

	int3 start = make_int3(sdr->start);
	int3 end = make_int3(sdr->end);

	int3 data_start = make_int3(sdr->full_buffer_start);
	int3 data_end = make_int3(sdr->full_buffer_end);

//	int halo = sdr->halo;

//    int X = end.x - start.x;
//    int Y = end.y - start.y;
//	int Z = end.z - start.z;

    float3 rbf = make_float3(0);

	//float x1 = p.x;
	//float y1 = p.y;
	//float z1 = p.z;

	//float x,y,z;
	//x = inv_modelview[0][0] * x1 + inv_modelview[0][1] * y1 + inv_modelview[0][2] * z1 + inv_modelview[0][3];
	//y = inv_modelview[1][0] * x1 + inv_modelview[1][1] * y1 + inv_modelview[1][2] * z1 + inv_modelview[1][3];
	//z = inv_modelview[2][0] * x1 + inv_modelview[2][1] * y1 + inv_modelview[2][2] * z1 + inv_modelview[2][3];

	//x = x - start.x;
	//y = y - start.y;
	//z = z - start.z;
//	int x, y, z;
//	x = p.x ;
//    y = p.y ;
//	z = p.z ;
    
	float3 alpha = 255 * (p - floor(p - 0.5f) - 0.5f);

	float3 hgx = hg(alpha.x);
	float3 hgy = hg(alpha.y);
	float3 hgz = hg(alpha.z);

	float3 dhgx = dhg(alpha.x);
	float3 dhgy = dhg(alpha.y);
	float3 dhgz = dhg(alpha.z);

	// compute x-derivative
	
	R q000 = linear_query_3d<R>(data, p.x - dhgx.x, p.y - hgy.x, p.z - hgz.x, sdr);
	R q001 = linear_query_3d<R>(data, p.x - dhgx.x, p.y - hgy.x, p.z + hgz.y, sdr);
	R q010 = linear_query_3d<R>(data, p.x - dhgx.x, p.y + hgy.y, p.z - hgz.x, sdr);
	R q011 = linear_query_3d<R>(data, p.x - dhgx.x, p.y + hgy.y, p.z + hgz.y, sdr);
	R q100 = linear_query_3d<R>(data, p.x + dhgx.y, p.y - hgy.x, p.z - hgz.x, sdr);
	R q101 = linear_query_3d<R>(data, p.x + dhgx.y, p.y - hgy.x, p.z + hgz.y, sdr);
	R q110 = linear_query_3d<R>(data, p.x + dhgx.y, p.y + hgy.y, p.z - hgz.x, sdr);
	R q111 = linear_query_3d<R>(data, p.x + dhgx.y, p.y + hgy.y, p.z + hgz.y, sdr);

	R q00 = lerp(q000, q001, hgz.z);
	R q01 = lerp(q010, q011, hgz.z);
	R q10 = lerp(q100, q101, hgz.z);
	R q11 = lerp(q110, q111, hgz.z);

	R q0 = lerp(q00, q01, hgy.z);
	R q1 = lerp(q10, q11, hgy.z);

	float gradientX = d_lerp(q0, q1, dhgx.z);

	// compute y-derivative
	q000 = linear_query_3d<R>(data, p.x - hgx.x, p.y - dhgy.x, p.z - hgz.x, sdr);
	q001 = linear_query_3d<R>(data, p.x - hgx.x, p.y - dhgy.x, p.z + hgz.y, sdr);
	q010 = linear_query_3d<R>(data, p.x - hgx.x, p.y + dhgy.y, p.z - hgz.x, sdr);
	q011 = linear_query_3d<R>(data, p.x - hgx.x, p.y + dhgy.y, p.z + hgz.y, sdr);
	q100 = linear_query_3d<R>(data, p.x + hgx.y, p.y - dhgy.x, p.z - hgz.x, sdr);
	q101 = linear_query_3d<R>(data, p.x + hgx.y, p.y - dhgy.x, p.z + hgz.y, sdr);
	q110 = linear_query_3d<R>(data, p.x + hgx.y, p.y + dhgy.y, p.z - hgz.x, sdr);
	q111 = linear_query_3d<R>(data, p.x + hgx.y, p.y + dhgy.y, p.z + hgz.y, sdr);

	q00 = lerp(q000, q001, hgz.z);
	q01 = lerp(q010, q011, hgz.z);
	q10 = lerp(q100, q101, hgz.z);
	q11 = lerp(q110, q111, hgz.z);

	q0 = d_lerp(q00, q01, dhgy.z);
	q1 = d_lerp(q10, q11, dhgy.z);

	float gradientY = lerp(q0, q1, hgx.z);

	// compute z-derivative
	q000 = linear_query_3d<R>(data, p.x - hgx.x, p.y - hgy.x, p.z - dhgz.x, sdr);
	q001 = linear_query_3d<R>(data, p.x - hgx.x, p.y - hgy.x, p.z + dhgz.y, sdr);
	q010 = linear_query_3d<R>(data, p.x - hgx.x, p.y + hgy.y, p.z - dhgz.x, sdr);
	q011 = linear_query_3d<R>(data, p.x - hgx.x, p.y + hgy.y, p.z + dhgz.y, sdr);
	q100 = linear_query_3d<R>(data, p.x + hgx.y, p.y - hgy.x, p.z - dhgz.x, sdr);
	q101 = linear_query_3d<R>(data, p.x + hgx.y, p.y - hgy.x, p.z + dhgz.y, sdr);
	q110 = linear_query_3d<R>(data, p.x + hgx.y, p.y + hgy.y, p.z - dhgz.x, sdr);
	q111 = linear_query_3d<R>(data, p.x + hgx.y, p.y + hgy.y, p.z + dhgz.y, sdr);

	q00 = d_lerp(q000, q001, dhgz.z);
	q01 = d_lerp(q010, q011, dhgz.z);
	q10 = d_lerp(q100, q101, dhgz.z);
	q11 = d_lerp(q110, q111, dhgz.z);

	q0 = lerp(q00, q01, hgy.z);
	q1 = lerp(q10, q11, hgy.z);

	float gradientZ = lerp(q0, q1, hgx.z);

	return make_float3(gradientX, gradientY, gradientZ);
}

//rotate functions
///////////////////////////////////////////////////////////////////////////////////
GPU float arccos(float angle){
	return acos(angle);
}
GPU float arcsin(float angle){
	return asin(angle);
}
GPU float norm(float3 a){
	float val = 0;
	val += a.x*a.x + a.y*a.y + a.z*a.z;
	val = sqrt(val);
	return val;
}

GPU float3 matmul(float3* mat, float3 vec){
    float x = mat[0].x*vec.x + mat[1].x*vec.y + mat[2].x*vec.z;
    float y = mat[0].y*vec.x + mat[1].y*vec.y + mat[2].y*vec.z;
    float z = mat[0].z*vec.x + mat[1].z*vec.y + mat[2].z*vec.z;

    return make_float3(x, y, z);
}
GPU void getInvMat(float3* mat, float3* ret) {
    double det = mat[0].x*(mat[1].y*mat[2].z-mat[1].z*mat[2].y)-mat[0].y*(mat[1].x*mat[2].z-mat[1].z*mat[2].x)+mat[0].z*(mat[1].x*mat[2].y-mat[1].y*mat[2].x);

    if(det!=0) {
        double invdet = 1/det;
        float a00 = (mat[1].y*mat[2].z-mat[2].y*mat[1].z)*invdet;
        float a01 = (mat[0].z*mat[2].y-mat[0].y*mat[2].z)*invdet;
        float a02 = (mat[0].y*mat[1].z-mat[0].z*mat[1].y)*invdet;
        float a10 = (mat[1].z*mat[2].x-mat[1].x*mat[2].z)*invdet;
        float a11 = (mat[0].x*mat[2].z-mat[0].z*mat[2].x)*invdet;
        float a12 = (mat[1].x*mat[0].z-mat[0].x*mat[1].z)*invdet;
        float a20 = (mat[1].x*mat[2].y-mat[2].x*mat[1].y)*invdet;
        float a21 = (mat[2].x*mat[0].y-mat[0].x*mat[2].y)*invdet;
        float a22 = (mat[0].x*mat[1].y-mat[1].x*mat[0].y)*invdet;
        ret[0] = make_float3(a00, a01, a02);
        ret[1] = make_float3(a10, a11, a12);
        ret[2] = make_float3(a20, a21, a22);
    }
    else {
        ret[0] = make_float3(0);
        ret[1] = make_float3(0);
        ret[2] = make_float3(0);
    }
}
GPU float getDistance(float3* mat, float3 vec){
    float3 tmp_mat  = matmul(mat, vec);
    if(tmp_mat.z>200000 ) return -8765;
    if(tmp_mat.z<0) tmp_mat.z = 0;
    if(tmp_mat.y < 0 || tmp_mat.y > 1) return -8765;
    if(tmp_mat.x < 0 || tmp_mat.x > 1) return -8765;

    if(tmp_mat.y+tmp_mat.x > 1.0000) return -8765;

    return tmp_mat.z;
}

GPU float2 getCrossedInterval(float3 origin, float3 direction, float3* tmp){
    int cnt = 0;

    float start=0, end=0;
    for(int i=1; i<8; i*=2) {
        if(cnt==2) break;
        float3 pos[3];
        pos[0] = tmp[0];
        pos[1] = tmp[i];
        int2 rest = (i==1)?make_int2(2, 4):((i==2)?make_int2(1,4):make_int2(1,2));
        for(int j=0; j<2; j++) {
            int target=(j==0)?rest.x:rest.y;
            pos[2] = tmp[i+target];

            float3 dist = pos[0] - origin;

            float3 tmp_pos[3];
            tmp_pos[0] = pos[0] - pos[1];
            tmp_pos[1] = pos[0] - pos[2];
            tmp_pos[2] = direction;
            float3 Norm = cross(tmp_pos[0],tmp_pos[1]);
            float dot_prod = dot(Norm, direction);
            if(dot_prod == 0) continue;

            float3 inv[3];
            getInvMat(tmp_pos,inv);
            float distance = getDistance(inv, dist);


            dist = pos[2] - origin;
            tmp_pos[0] = pos[2] - pos[0];
            tmp_pos[1] = pos[2] - pos[1];

            getInvMat(tmp_pos,inv);

            float distance2 = getDistance(inv, dist);
            if(distance2 != distance)
                distance=(distance2 == start)?distance:distance2;


            if(distance != -8765) {
                if(cnt==0) {
                    start = distance;
                    cnt++;
                }
                else if(cnt==1) {
                    if(fabs(start - distance) < 0.001f) break;
                    end   = (distance > start)? distance : start;
                    start = (distance > start)? start : distance;
                    cnt++;
                }
            }
        }

    }
  for(int i=1; i<8; i*=2) {
        if(cnt==2)  break;
        float3 pos[3];
        pos[0] = tmp[7];
        pos[1] = tmp[7-i];
        int2 rest = (i==1)?make_int2(2, 4):((i==2)?make_int2(1,4):make_int2(1,2));
        for(int j=0; j<2; j++) {
            int target=(j==0)?rest.x:rest.y;
            pos[2] = tmp[7-i-target];

            float3 dist = pos[0] - origin;

            float3 tmp_pos[3];
            tmp_pos[0] = pos[0] - pos[1];
            tmp_pos[1] = pos[0] - pos[2];
            tmp_pos[2] = direction;
            float3 Norm = cross(tmp_pos[0],tmp_pos[1]);
            float dot_prod = dot(Norm, direction);
            if(dot_prod == 0) continue;

            float3 inv[3];
            getInvMat(tmp_pos,inv);

            float distance = getDistance(inv, dist);

            dist = pos[2] - origin;
            tmp_pos[0] = pos[2] - pos[0];
            tmp_pos[1] = pos[2] - pos[1];

            getInvMat(tmp_pos,inv);

            float distance2 = getDistance(inv, dist);
            if(distance2 != distance)
                distance = (distance2 == start)?distance:distance2;

            if(distance != -8765) {
                if(cnt==0) {
                    start = distance;
                    cnt++;
                }
                else if(cnt==1) {
                    if(fabs(start - distance)<0.001f) break;
                    end   = (distance > start)?distance : start;
                    start = (distance > start)?start : distance;
                    cnt++;
                }
            }
        }
    }
    if (start > end)
        end = start;

    return make_float2(start, end);
}

template<typename T> GPU line_iter perspective_iter(T* volume, float x, float y, float step, float near, VIVALDI_DATA_RANGE* sdr){
	int3 start = make_int3(sdr->start);
	int3 end = make_int3(sdr->end);

	int3 data_start = make_int3(sdr->full_buffer_start);
	int3 data_end = make_int3(sdr->full_buffer_end);


	float3 ray_direction = make_float3(x,y,near);
	float3 ray_origin = make_float3(0);

	float3 tmp_cubic[8];

	for(int i = 0 ; i < 2 ; i ++){
		float xp;
		if(i == 0)xp = start.x;
		else xp = end.x;
		for(int j = 0 ; j < 2; j ++){
			float yp;
			if(j == 0)yp = start.y;
			else yp = end.y;
			for(int k = 0 ; k < 2; k ++){
				float zp;
				if(k == 0)zp = start.z;
				else zp = end.z;
				float xt = modelview[0][0] * xp + modelview[0][1] * yp + modelview[0][2] * zp + modelview[0][3];
				float yt = modelview[1][0] * xp + modelview[1][1] * yp + modelview[1][2] * zp + modelview[1][3];
				float zt = modelview[2][0] * xp + modelview[2][1] * yp + modelview[2][2] * zp + modelview[2][3];

				tmp_cubic[4*i + 2*j + k] = make_float3(xt, yt, zt);
			}
		}
	}


	float2 interval = getCrossedInterval(ray_origin, ray_direction, tmp_cubic);

	float3 S = ray_origin + interval.x * ray_direction;
	float3 E = ray_origin + interval.y * ray_direction;	
    float len = 1;

    if (interval.x !=interval.y)  {
        len = length(S-E);
        float xx,yy,zz;

        xx = inv_modelview[0][0] * S.x + inv_modelview[0][1] * S.y + inv_modelview[0][2] * S.z + inv_modelview[0][3];
        yy = inv_modelview[1][0] * S.x + inv_modelview[1][1] * S.y + inv_modelview[1][2] * S.z + inv_modelview[1][3];
        zz = inv_modelview[2][0] * S.x + inv_modelview[2][1] * S.y + inv_modelview[2][2] * S.z + inv_modelview[2][3];
        S.x = xx - start.x;
        S.y = yy - start.y; 
        S.z = zz - start.z; 
    
    
        xx = inv_modelview[0][0] * E.x + inv_modelview[0][1] * E.y + inv_modelview[0][2] * E.z + inv_modelview[0][3];
        yy = inv_modelview[1][0] * E.x + inv_modelview[1][1] * E.y + inv_modelview[1][2] * E.z + inv_modelview[1][3];
        zz = inv_modelview[2][0] * E.x + inv_modelview[2][1] * E.y + inv_modelview[2][2] * E.z + inv_modelview[2][3];
        E.x = xx - start.x;
        E.y = yy - start.y;
        E.z = zz - start.z;
    }
	return line_iter(S,E,step*length(S-E)/len);
}
template<typename T> GPU line_iter orthogonal_iter(T* volume, float2 p, float step, VIVALDI_DATA_RANGE* sdr){

	// initialization
    int3 start = make_int3(sdr->start);
    int3 end = make_int3(sdr->end);

    int3 data_start = make_int3(sdr->full_buffer_start);
    int3 data_end = make_int3(sdr->full_buffer_end);

	int buffer_halo = sdr->buffer_halo;
	
    float3 ray_direction = make_float3(0,0,1);
    float3 ray_origin = make_float3(p.x, p.y ,0);

	start = start +  make_int3(buffer_halo);
	end = end - make_int3(buffer_halo);


	
    float3 tmp_cubic[8];


    for(int i = 0 ; i < 2 ; i ++){
        float xp;
        if(i == 0)xp = start.x;
        else xp = end.x;
        for(int j = 0 ; j < 2; j ++){
            float yp;
            if(j == 0)yp = start.y;
            else yp = end.y;
            for(int k = 0 ; k < 2; k ++){
                float zp;
                if(k == 0)zp = start.z;
                else zp = end.z;
                float xt = modelview[0][0] * xp + modelview[0][1] * yp + modelview[0][2] * zp + modelview[0][3];
                float yt = modelview[1][0] * xp + modelview[1][1] * yp + modelview[1][2] * zp + modelview[1][3];
                float zt = modelview[2][0] * xp + modelview[2][1] * yp + modelview[2][2] * zp + modelview[2][3];
                
                tmp_cubic[4*i + 2*j + k] = make_float3(xt, yt, zt);
            }
        }
    }

    float2 interval = getCrossedInterval(ray_origin, ray_direction, tmp_cubic);

    
    float3 S = ray_origin + interval.x * ray_direction;

    float3 E = ray_origin + interval.y * ray_direction; 
    float len = 1;

	start = start - make_int3(buffer_halo);
    if (interval.x !=interval.y)  {
        len = length(S-E);
        float xx,yy,zz;

        xx = inv_modelview[0][0] * S.x + inv_modelview[0][1] * S.y + inv_modelview[0][2] * S.z + inv_modelview[0][3];
        yy = inv_modelview[1][0] * S.x + inv_modelview[1][1] * S.y + inv_modelview[1][2] * S.z + inv_modelview[1][3];
        zz = inv_modelview[2][0] * S.x + inv_modelview[2][1] * S.y + inv_modelview[2][2] * S.z + inv_modelview[2][3];
        S.x = xx - start.x;
        S.y = yy - start.y; 
        S.z = zz - start.z; 
    
    
        xx = inv_modelview[0][0] * E.x + inv_modelview[0][1] * E.y + inv_modelview[0][2] * E.z + inv_modelview[0][3];
        yy = inv_modelview[1][0] * E.x + inv_modelview[1][1] * E.y + inv_modelview[1][2] * E.z + inv_modelview[1][3];
        zz = inv_modelview[2][0] * E.x + inv_modelview[2][1] * E.y + inv_modelview[2][2] * E.z + inv_modelview[2][3];
        E.x = xx - start.x;
        E.y = yy - start.y;
        E.z = zz - start.z;
    }
	return line_iter(S,E,step*length(S-E)/len);
}
template<typename T> GPU line_iter orthogonal_iter(T* volume, float x, float y, float step, VIVALDI_DATA_RANGE* sdr){
	return orthogonal_iter(volume, make_float2(x,y), step, sdr);
}

// Domain Speicifc functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GPU float3 phong(float3 L, float3 pos, float3 N, float3 omega, float3 kd, float3 ks, float n, float3 amb){
	float3 color;
	float a, b, c;

    a = inv_modelview[0][0] * L.x + inv_modelview[0][1] * L.y + inv_modelview[0][2] * L.z + inv_modelview[0][3];
    b = inv_modelview[1][0] * L.x + inv_modelview[1][1] * L.y + inv_modelview[1][2] * L.z + inv_modelview[1][3];
    c = inv_modelview[2][0] * L.x + inv_modelview[2][1] * L.y + inv_modelview[2][2] * L.z + inv_modelview[2][3];
	L.x = a;
	L.y = b;
	L.z = c;
	L = normalize(L-pos);
	//ambient
	color = amb;

	// diffuse
    float lobe =  max(dot(N, L), 0.0f);
    
	color += kd * lobe;

    // specular
    if (n > 0)
    {
        float3 R = reflect(-L, N);
        lobe = pow(fmaxf(dot(R, omega), 0), n);
        color += ks * lobe;
    }
 
    // clamping is a hack, but looks better
    return fminf(color, make_float3(1));
}
GPU float3 phong(float3 L, float3 N, float3 omega, float3 kd, float3 ks, float n, float3 amb){
	float3 color;
	
	//ambient
	color = amb;

	// diffuse
    float lobe =  max(dot(N, L), 0.0f);
    
	color += kd * lobe;

    // specular
    if (n > 0)
    {
        float3 R = reflect(-L, N);
        lobe = pow(fmaxf(dot(R, omega), 0), n);
        color += ks * lobe;
    }
 
    // clamping is a hack, but looks better
    return fminf(color, make_float3(1));
}
GPU float3 diffuse(float3 L, float3 N, float3 kd){
    float lobe = max(dot(N, L), 0.0f);
	return kd * lobe;
}
template<typename R,typename T> GPU R laplacian(T* image, float2 p, VIVALDI_DATA_RANGE* sdr){

	float x = p.x;
	float y = p.y;

	//parallel variables
	R a =  point_query_2d<R>(image, x, y, sdr);
	R u =  point_query_2d<R>(image, x, y+1, sdr);
	R d =  point_query_2d<R>(image, x, y-1, sdr);
	R l =  point_query_2d<R>(image, x-1, y, sdr);
	R r =  point_query_2d<R>(image, x+1, y, sdr);
	R ret = u+d+l+r-4.0*a;
	return ret;

}
template<typename R,typename T> GPU R laplacian(T* image, float x, float y, VIVALDI_DATA_RANGE* sdr){

	//parallel variables
	return laplacian<R>(image, make_float2(x,y), sdr);

}





extern "C"{
// memory copy functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// 2d copy functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void writing_2d( float* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            float* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;

        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);

        float a = point_query_2d<float>(A, x, y, A_DATA_RANGE);
        rb[idx] = a;


    }

__global__ void writing_2d_uchar( uchar* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            uchar* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;
        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        float val = point_query_2d<float>(A, x, y, A_DATA_RANGE);
        uchar a = (unsigned char)val;
        rb[idx] = a;
  }


__global__ void writing_2d_uchar2( uchar2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            uchar2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;

        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        float2 val = point_query_2d<float2>(A, x, y, A_DATA_RANGE);
        uchar2 a = make_uchar2(val);
        rb[idx] = a;
  }


__global__ void writing_2d_uchar3( uchar3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            uchar3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;
        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        float3 val = point_query_2d<float3>(A, x, y, A_DATA_RANGE);
        uchar3 a = make_uchar3(val);
        rb[idx] = a;
  }


__global__ void writing_2d_uchar4( uchar4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            uchar4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;

        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        float4 val = point_query_2d<float4>(A, x, y, A_DATA_RANGE);
        uchar4 a = make_uchar4(val);
        rb[idx] = a;
  }



__global__ void writing_2d_short( short* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            short* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;
        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        float val = point_query_2d<float>(A, x, y, A_DATA_RANGE);
        short a = short(val);
        rb[idx] = a;
  }

__global__ void writing_2d_short2( short2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            short2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;
        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        float2 val = point_query_2d<float2>(A, x, y, A_DATA_RANGE);
        short2 a = make_short2(val);
        rb[idx] = a;
  }


__global__ void writing_2d_short3( short3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            short3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;
        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        float3 val = point_query_2d<float3>(A, x, y, A_DATA_RANGE);
        short3 a = make_short3(val);
        rb[idx] = a;
  }

__global__ void writing_2d_short4( short4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            short4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;
        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        float4 val = point_query_2d<float4>(A, x, y, A_DATA_RANGE);
        short4 a = make_short4(val);
        rb[idx] = a;
  }





__global__ void writing_2d_RGBA( RGBA* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            RGBA* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;

        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        RGBA a = RGBA(point_query_2d<float4>(A, x, y, A_DATA_RANGE));
        rb[idx] = a;
  }



__global__ void writing_2d_RGB( RGB* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            RGB* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;

        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        RGB a = RGB(point_query_2d<float3>(A, x, y, A_DATA_RANGE));
        rb[idx] = a;
  }

  __global__ void writing_2d_int(int* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            int* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;
        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        float a = point_query_2d<float>(A, x, y, A_DATA_RANGE);
        rb[idx] = int(a);
  }

__global__ void writing_2d_int2(int2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            int2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;
        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        float2 a = point_query_2d<float2>(A, x, y, A_DATA_RANGE);
        rb[idx] = make_int2(a);
  }


__global__ void writing_2d_int3( int3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            int3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;

        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        float3 a = point_query_2d<float3>(A, x, y, A_DATA_RANGE);
        rb[idx] = make_int3(a);
  }

__global__ void writing_2d_int4(int4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            int4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;
        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        float4 a = point_query_2d<float4>(A, x, y, A_DATA_RANGE);
        rb[idx] = make_int4(a);
  }


__global__ void writing_2d_float( float* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            float* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;
        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        float a = point_query_2d<float>(A, x, y, A_DATA_RANGE);
        rb[idx] = a;
  }

__global__ void writing_2d_float2( float2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            float2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;
        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        float2 a = point_query_2d<float2>(A, x, y, A_DATA_RANGE);
        rb[idx] = a;
  }


__global__ void writing_2d_float3( float3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            float3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;

        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        float3 a = point_query_2d<float3>(A, x, y, A_DATA_RANGE);
        rb[idx] = a;
  }

__global__ void writing_2d_float4( float4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
                            float4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
                            int y_start, int y_end, int x_start, int x_end)
    {
        //parallel variables
        int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
        int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;

        int x = x_hschoi + x_start;
        int y = y_hschoi + y_start;

        if(x_end <= x || y_end <= y)return;
        int idx = (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x);
        float4 a = point_query_2d<float4>(A, x, y, A_DATA_RANGE);
        rb[idx] = a;
  }




// 3d copy functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// halo memset function
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
__global__ void halo_memeset( float3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
								int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
            {
                //parallel variables
                int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
                int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
                int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;

                int x = x_hschoi + x_start;
                int y = y_hschoi + y_start;
                int z = z_hschoi + z_start;

                if(x_end <= x || y_end <= y || z_end <= z)return;
                int idx =
                (z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
                + (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
                + (x-rb_DATA_RANGE->start.x);


				int3 data_start = make_int3(rb_DATA_RANGE->full_buffer_start);
				int3 data_end = make_int3(rb_DATA_RANGE->full_buffer_end);

				int3 start = make_int3(rb_DATA_RANGE->start);
				int3 end = make_int3(rb_DATA_RANGE->end);

				x = x - start.x;
				y = y - start.y;
				z = z - start.z;
				int buffer_X = end.x - start.x;
				int buffer_Y = end.y - start.y;
				int buffer_Z = end.z - start.z;

				if (!(data_start.x <= x && x < data_end.x &&
					  data_start.y <= y && y < data_end.y &&
	  				  data_start.y <= y && y < data_end.z)){
					r[idx] = initial(rb[0]);
				}
            }

}
*/
}

__device__ float4 alpha_compositing(float4 origin, float4 next)
{
	float a = origin.w;
	float r = origin.x;
	float g = origin.y;
	float b = origin.z;
	
	float x, y, z, w;
	w = a + (1-a/255) * next.w;
	
	w = (w > 255)? 255 : w;
	
	x = r + (1-a/255) * next.x * next.w/255.0f;
	y = g + (1-a/255) * next.y * next.w/255.0f;
	z = b + (1-a/255) * next.z * next.w/255.0f;

	return make_float4(x,y,z,w);
}

__device__ float4 alpha_compositing_wo_alpha(float4 origin, float4 next)
{
        float a = origin.w;
        float r = origin.x;
        float g = origin.y;
        float b = origin.z;
	//if(origin.w == 1) return make_float4(255,0,0,0);

        float x, y, z, w;
	w = a + (1-a/255) * next.w;

	w = (w > 255)? 255 : w;

    x = r + (1-a/255) * next.x;
    y = g + (1-a/255) * next.y;
    z = b + (1-a/255) * next.z;

	return make_float4(x,y,z,w);
}
__device__ float4 background_white(float4 origin)
{
	float a = origin.w;
	float r = origin.x;
	float g = origin.y;
	float b = origin.z;

	float x, y, z, w;
	w = a ;

	x = r + (1-a/255.0f) * 255.0f;
	y = g + (1-a/255.0f) * 255.0f;
	z = b + (1-a/255.0f) * 255.0f;

	return make_float4(x,y,z,w);
}

__device__ float4 detach(float4 origin)
{
	float a = origin.w;
	float r = origin.x;
	float g = origin.y;
	float b = origin.z;

	float x,y,z,w;
	w = a;
	x = r - (1-a/255.0)*255.0;
	y = g - (1-a/255.0)*255.0;
	z = b - (1-a/255.0)*255.0;

	return make_float4(x,y,z,w);
}


texture<float4, 2> TFF;

#include <stdio.h>
__device__ float4 transfer(float a)
{
	float4 tmp = tex2D(TFF, a/TF_bandwidth* 255, 0);
	float4 tmp_col = make_float4(tmp.x*255.0, tmp.y*255.0, tmp.z*255.0, tmp.w*255);


	return tmp_col;
}
__device__ float4 transfer(float2 a)
{
	return transfer(a.x);
}
__device__ float4 transfer(float3 a)
{
	return transfer(a.x);
}
__device__ float4 transfer(float4 a)
{
	return transfer(a.x);
}
texture<float4, 2> TFF1;
texture<float4, 2> TFF2;
texture<float4, 2> TFF3;
texture<float4, 2> TFF4;
__device__ float4 transfer(float a, int chan)
{
	float4 tmp;
	if(chan == 0) 
		tmp = tex2D(TFF, a/TF_bandwidth * 255, 0);
	else if(chan == 1) 
		tmp = tex2D(TFF1, a/TF_bandwidth * 255, 0);
	else if(chan == 2) 
		tmp = tex2D(TFF2, a/TF_bandwidth * 255, 0);
	else if(chan == 3) 
		tmp = tex2D(TFF3, a/TF_bandwidth * 255, 0);
	else if(chan == 4) 
		tmp = tex2D(TFF4, a/TF_bandwidth * 255, 0);

	float4 tmp_col = make_float4(tmp.x*255.0, tmp.y*255.0, tmp.z*255.0, tmp.w*255);

	return tmp_col;
}
__device__ float4 transfer(float2 a, int chan)
{
	return transfer(a.x, chan);
}
__device__ float4 transfer(float3 a, int chan)
{
	return transfer(a.x, chan);
}
__device__ float4 transfer(float4 a, int chan)
{
	return transfer(a.x, chan);
}
__device__ float4 ch_binder(float4 a, float4 b)
{
	return (a + b) / 2.0f;
}
__device__ int floor_tmp(float a)
{
	return floor(a);
}
extern "C"{

	__global__ void writing_3d_uchar(uchar* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uchar* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
		+ (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
		+ (x-rb_DATA_RANGE->start.x);
		
		float val = point_query_3d<float>(A, x, y, z, A_DATA_RANGE);
		uchar a = (unsigned char)(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_uchar2(uchar2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uchar2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
		+ (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
		+ (x-rb_DATA_RANGE->start.x);
		
		float2 val = point_query_3d<float2>(A, x, y, z, A_DATA_RANGE);
		uchar2 a = make_uchar2(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_uchar3(uchar3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uchar3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
		+ (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
		+ (x-rb_DATA_RANGE->start.x);
		
		float3 val = point_query_3d<float3>(A, x, y, z, A_DATA_RANGE);
		uchar3 a = make_uchar3(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_uchar4(uchar4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uchar4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
		+ (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
		+ (x-rb_DATA_RANGE->start.x);
		
		float4 val = point_query_3d<float4>(A, x, y, z, A_DATA_RANGE);
		uchar4 a = make_uchar4(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_short(short* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	short* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
		+ (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
		+ (x-rb_DATA_RANGE->start.x);
		
		float val = point_query_3d<float>(A, x, y, z, A_DATA_RANGE);
		short a = short(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_short2(short2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	short2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
		+ (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
		+ (x-rb_DATA_RANGE->start.x);
		
		float2 val = point_query_3d<float2>(A, x, y, z, A_DATA_RANGE);
		short2 a = make_short2(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_short3(short3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	short3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
		+ (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
		+ (x-rb_DATA_RANGE->start.x);
		
		float3 val = point_query_3d<float3>(A, x, y, z, A_DATA_RANGE);
		short3 a = make_short3(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_short4(short4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	short4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
		+ (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
		+ (x-rb_DATA_RANGE->start.x);
		
		float4 val = point_query_3d<float4>(A, x, y, z, A_DATA_RANGE);
		short4 a = make_short4(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_int(int* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	int* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
		+ (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
		+ (x-rb_DATA_RANGE->start.x);
		
		float val = point_query_3d<float>(A, x, y, z, A_DATA_RANGE);
		int a = int(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_int2(int2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	int2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
		+ (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
		+ (x-rb_DATA_RANGE->start.x);
		
		float2 val = point_query_3d<float2>(A, x, y, z, A_DATA_RANGE);
		int2 a = make_int2(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_int3(int3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	int3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
		+ (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
		+ (x-rb_DATA_RANGE->start.x);
		
		float3 val = point_query_3d<float3>(A, x, y, z, A_DATA_RANGE);
		int3 a = make_int3(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_int4(int4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	int4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
		+ (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
		+ (x-rb_DATA_RANGE->start.x);
		
		float4 val = point_query_3d<float4>(A, x, y, z, A_DATA_RANGE);
		int4 a = make_int4(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_float(float* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	float* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
		+ (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
		+ (x-rb_DATA_RANGE->start.x);
		
		float val = point_query_3d<float>(A, x, y, z, A_DATA_RANGE);
		float a = float(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_float2(float2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	float2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
		+ (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
		+ (x-rb_DATA_RANGE->start.x);
		
		float2 val = point_query_3d<float2>(A, x, y, z, A_DATA_RANGE);
		float2 a = float2(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_float3(float3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	float3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
		+ (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
		+ (x-rb_DATA_RANGE->start.x);
		
		float3 val = point_query_3d<float3>(A, x, y, z, A_DATA_RANGE);
		float3 a = float3(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_float4(float4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	float4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
		+ (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
		+ (x-rb_DATA_RANGE->start.x);
		
		float4 val = point_query_3d<float4>(A, x, y, z, A_DATA_RANGE);
		float4 a = float4(val);
		rb[idx] = a;
	}
						
}

__global__ void mipshort4(int* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE, short4* volume, VIVALDI_DATA_RANGE* volume_DATA_RANGE, int x_start, int x_end, int y_start, int y_end){

    int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
    int x = x_start + x_hschoi;
    int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
    int y = y_start + y_hschoi;

    if(x_end <= x || y_end <= y)return;
    line_iter line_iter;
    float step;
    int max;
    step = 1.0;
    line_iter = orthogonal_iter(volume, x, y, step, volume_DATA_RANGE);

    max = 0;
    for(float3 elem = line_iter.begin(); line_iter.hasNext(); ){
        float4 val;
        val = linear_query_3d<float4>(volume, elem, volume_DATA_RANGE);
        if( max < val){
             max = val;
         }
        elem = line_iter.next();
    }
    rb[(x-rb_DATA_RANGE->start.x)+(y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)] = max;
    return;

}


			int main()
			{
					return EXIT_SUCCESS;
			}
			