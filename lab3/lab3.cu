#include "lab3.h"
#include <cstdio>

//#define NUM_OF_ITERATION 	100
#define NUM_OF_ITERATION 	6000
#define CROSS_BORDER 		-1

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
#if 1	// debug
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
#endif
	}
}

__global__ void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int thread_y = blockIdx.y * blockDim.y + threadIdx.y;

	const int Ct = wt * thread_y + thread_x ;
	if( thread_y < ht && thread_x < wt && mask[Ct] > 127.0){
		int Nt = (thread_y > 0) ?		Ct-wt : CROSS_BORDER ;
		int Wt = (thread_x > 0) ? 	Ct-1  : CROSS_BORDER ;
		int St = (thread_y < ht-1) ? 	Ct+wt : CROSS_BORDER ;
		int Et = (thread_x < wt-1) ? 	Ct+1  : CROSS_BORDER ;

		fixed[Ct*3  ] = 4*target[Ct*3  ] ;
		fixed[Ct*3+1] = 4*target[Ct*3+1] ;
		fixed[Ct*3+2] = 4*target[Ct*3+2] ;

		for (int i=0; i<3; i++) { 
			if ( Nt != CROSS_BORDER ) fixed[Ct*3+i] -= target[Nt*3+i] ;
			else fixed[Ct*3+i] -= target[Ct*3+i] ;

			if ( Wt != CROSS_BORDER ) fixed[Ct*3+i] -= target[Wt*3+i] ;
			else fixed[Ct*3+i] -= target[Ct*3+i] ;

			if ( St != CROSS_BORDER ) fixed[Ct*3+i] -= target[St*3+i] ;
			else fixed[Ct*3+i] -= target[Ct*3+i] ;

			if ( Et != CROSS_BORDER ) fixed[Ct*3+i] -= target[Et*3+i] ;
			else fixed[Ct*3+i] -= target[Ct*3+i] ;
		}

		int Yb = oy + thread_y ;
		int Xb = ox + thread_x ;
		if( 0 <= Yb && Yb < hb && 0 <= Xb && Xb < wb){
			int Cb = wb * Yb + Xb ;
			
			int Nb = (Yb > 0) ? 	Cb-wb : Cb ;
			int Wb = (Xb > 0) ? 	Cb-1  : Cb ;
			int Sb = (Yb < hb-1) ? 	Cb+wb : Cb ;
			int Eb = (Xb < wb-1) ? 	Cb+1  : Cb ;

			for (int j=0; j<3; j++) { 
				if( Nt == CROSS_BORDER || mask[Nt] <= 127.0) fixed[Ct*3+j] += background[Nb*3+j] ; 
				if( Wt == CROSS_BORDER || mask[Wt] <= 127.0) fixed[Ct*3+j] += background[Wb*3+j] ; 
				if( St == CROSS_BORDER || mask[St] <= 127.0) fixed[Ct*3+j] += background[Sb*3+j] ; 
				if( Et == CROSS_BORDER || mask[Et] <= 127.0) fixed[Ct*3+j] += background[Eb*3+j] ; 
			}
		}
	}
}

__global__ void PoissonImageCloningIteration(
	const float *fixed, 
	const float *mask, 
	const float *buf1, 
	float *buf2, 
	const int width_t, const int height_t
)
{
#if 1	// debug
	const int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int thread_y = blockIdx.y * blockDim.y + threadIdx.y;

	const int Ct = width_t * thread_y + thread_x ;

	if( thread_y < height_t and thread_x < width_t and mask[Ct] > 127.0f){
		int Nt = (thread_y > 0) ? 		Ct-width_t : CROSS_BORDER ;
		int Wt = (thread_x > 0) ? 		Ct-1	   : CROSS_BORDER ;
		int St = (thread_y < height_t-1) ? 	Ct+width_t : CROSS_BORDER ;
		int Et = (thread_x < width_t-1) ? 	Ct+1 	   : CROSS_BORDER ;

		buf2[Ct*3  ] = fixed[Ct*3] ;
		buf2[Ct*3+1] = fixed[Ct*3+1] ;
		buf2[Ct*3+2] = fixed[Ct*3+2] ;

		for (int i=0; i<3; i++) { 
			if(Nt != CROSS_BORDER && mask[Nt] > 127.0f ) buf2[Ct*3+i] += buf1[Nt*3+i] ; 
			if(Wt != CROSS_BORDER && mask[Wt] > 127.0f ) buf2[Ct*3+i] += buf1[Wt*3+i] ; 
			if(St != CROSS_BORDER && mask[St] > 127.0f ) buf2[Ct*3+i] += buf1[St*3+i] ; 
			if(Et != CROSS_BORDER && mask[Et] > 127.0f ) buf2[Ct*3+i] += buf1[Et*3+i] ; 
		}

		buf2[Ct*3  ] /= 4 ;
		buf2[Ct*3+1] /= 4 ;
		buf2[Ct*3+2] /= 4 ;
	}
#endif
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{

#if 1	// debug
        // set up
        float *fixed, *buf1, *buf2 ;
        cudaMalloc(&fixed, 3*wt*ht*sizeof(float)) ;
        cudaMalloc(&buf1, 3*wt*ht*sizeof(float)) ;
        cudaMalloc(&buf2, 3*wt*ht*sizeof(float)) ;

        // initialize the iteration
        dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16) ;
        CalculateFixed<<<gdim, bdim>>> (
           background, target, mask, fixed,
           wb, hb, wt, ht, oy, ox
        ) ;

        //cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice) ;

	// iterate
        for (int t = 0 ; t < NUM_OF_ITERATION ; ++t) {
           PoissonImageCloningIteration<<<gdim, bdim>>>(
              fixed, mask, buf1, buf2, wt, ht
           );
           PoissonImageCloningIteration<<<gdim, bdim>>>(
              fixed, mask, buf2, buf1, wt, ht
           );
        }

       	// copy the image back 
        cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
        SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
                background, buf1, mask, output,
                wb, hb, wt, ht, oy, ox
        ) ;

        // clean up
        cudaFree(fixed) ;
        cudaFree(buf1) ;
        cudaFree(buf2) ;
#else
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, target, mask, output,
		wb, hb, wt, ht, oy, ox
	);
#endif
}
