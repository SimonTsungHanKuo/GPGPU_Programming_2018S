#include "lab1.h"
#include "SyncedMemory.h"

static const int W = 640 ;
static const int H = 480 ;
static const int NFRAME = 360 ;

//static const float julia_re = 0.28 ;  
//static const float julia_im = 0.008 ; 
static const float julia_re = -0.737 ;  
static const float julia_im = -0.208 ; 
static const float julia_re_scale = 1.8 ;
static const float julia_im_scale = 1.8 ;
static const float julia_zoom = 0.01 ;
static const float julia_speed = 0.5 ;
static const int   julia_max_iter = 100 ;

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

struct Lab1VideoGenerator::Impl {
	int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};

struct complex_num {
	float re ;
	float im ;

	__host__ __device__ complex_num(float _re, float _im): re(_re), im(_im) {}

	__device__ float len_sqr(){
		return re*re + im*im ;
	}

	__device__ complex_num operator*(const complex_num &c){
		return complex_num(re*c.re - im*c.im, im*c.re + re*c.im) ;
	}

	__device__ complex_num operator*(const int x){
		return complex_num(re*x, im*x) ;
	}

	__device__ complex_num operator+(const complex_num &c){
		return complex_num(re+c.re, im+c.im) ;
	}

	__device__ complex_num operator-(const complex_num &c){
		return complex_num(re-c.re, im-c.im) ;
	}
} ;

__device__ int find_julia(int x, int y, float zoom, int iter){

	float new_x = julia_re_scale * (float)(W/2-x)/(W/2*zoom) ;
	float new_y = julia_im_scale * (float)(H/2-y)/(H/2*zoom) ;

	complex_num c(julia_re, julia_im);
	complex_num z(new_x, new_y) ;

	for(int i = 0 ; i < iter ; ++i){
		z = z * z + c ;
		if( z.len_sqr() > julia_max_iter )
			return 0;
	}

	return 1 ;
}

__global__ void Draw(uint8_t *yuv, const int t){
	const int x = blockIdx.x * blockDim.x + threadIdx.x ;
	const int y = blockIdx.y * blockDim.y + threadIdx.y ;
	
	//int draw = find_julia(x, y, pow(1+julia_zoom, t), t*julia_speed) ;		// Normal
	int draw = find_julia(x, y, pow(1+julia_zoom, 2), t*julia_speed) ;		// Popular, tender 
	//int draw = find_julia(x, y, sin(pow(1+julia_zoom, t*3/4)), t*julia_speed) ;	// Final, I like this wild 
	//int draw = find_julia(x, y, sin(pow(1+julia_zoom, 2*t)), t*julia_speed) ;	// High Frequency

	int R = (float)(abs(2 * t - NFRAME))/NFRAME * 240 * draw ;
	int G = ( 1 - abs( 1 - 2 * abs(2 * (float)(t)/NFRAME - 1) ) ) * 240 * draw ;
	int B = ( 1 - (float)(abs(2 * t - NFRAME))/NFRAME ) * 240 * draw ;

	int Y =  0.299*R + 0.587*G + 0.114*B ;
	int U = -0.169*R - 0.332*G + 0.500*B + 128 ;
	int V =  0.500*R - 0.419*G - 0.081*B + 128 ;

	if( x < W and y < H ){
		yuv[y*W+x] = Y ;
		yuv[W*H + (y/2) * (W/2) + (x/2)] = U ;
		yuv[W*H*5/4 + (y/2) * (W/2) + (x/2)] = V ;
	}
}

void Lab1VideoGenerator::Generate(uint8_t *yuv){
	Draw<<<dim3((W-1)/32+1,(H-1)/32+1), dim3(32,32)>>>(yuv, impl->t) ;
	CHECK ;
	++(impl->t) ;
}

