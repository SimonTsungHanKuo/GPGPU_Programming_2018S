#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

//-------------------------------Below is Part I ---------------------------------------------------------
template <typename T>
struct set_zero : public thrust::unary_function<T, int>
{
	__host__ __device__
  	int operator()(T x) const {return 0;}
};

struct is_new_line
{
  __host__ __device__
  bool operator()(char x) {return x=='\n';}
};

void CountPosition1(const char *text, int *pos, int text_size)
{
	//std::cout << "text_size: " << text_size << std::endl; 
	
	thrust::device_ptr<const char> 	device_text(text) ;
        thrust::device_ptr<int>		device_pos(pos) ;
        thrust::fill(device_pos, device_pos+text_size, 1);

        is_new_line 	pred;
        set_zero<int>	op;
	thrust::transform_if(thrust::device, device_text, device_text + text_size, device_pos, op, pred) ;
	thrust::inclusive_scan_by_key(device_pos, device_pos + text_size, device_pos, device_pos);
	
#if 0
	using namespace std;
	thrust::copy(device_text, device_text+text_size, std::ostream_iterator<char>(std::cout, " "));
	cout << endl << "-------------------------------------------------------------------" << endl;
	thrust::copy(device_pos, device_pos+text_size, std::ostream_iterator<int>(std::cout, " "));
#endif
}

//-------------------------------Below is Part II --------------------------------------------------------

#define N 	(2*2048)	/*In TX2, # of threads per MP = 2048, ther are 2 MP  */
#define TPB 	(1024)		/*In TX2, # of threads per bock = 1024 */

__global__ void my_transform_if (const char *text, int *pos, int text_size) 
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x ;
	if (idx >= text_size) return;
	
	if (text[idx] == '\n') {
		pos[idx] = 0;
	} else {  
		pos[idx] = 1;
	}
	
	return ;
}

__global__ void my_inclusive_scan_by_key(int *pos, int text_size) 
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x ;
	if (idx >= text_size) return;
	if (pos[idx] == 0) goto exit;
	if (idx-1>=0 && pos[idx-1] == 1) goto exit ; 

	while (idx < text_size && pos[idx] !=0 ) {
		pos[idx] += pos[idx-1];
		idx++;
	}
exit:
	__syncthreads ();
	return ;
}

void CountPosition2(const char *text, int *pos, int text_size)
{
	int Q = text_size / N; 
	int R = text_size % N;
	int i = 0;
	// printf ("text_size: %d Q: %d R: %d \n", text_size, Q, R); 
	// do transform_if - set '\n' corresponding pos to be zero
	for (i=0; i<Q; i++) 
		my_transform_if <<<N/TPB, TPB>>> (text+i*N, pos+i*N, N);

	if  (R>0) 
		my_transform_if <<<N/TPB, TPB>>> (text+N*Q, pos+N*Q, R);

	Q = (text_size-1) / N;
	R = (text_size-1) % N;
	// printf ("text_size: %d Q: %d R: %d \n", (text_size-1), Q, R); 
	// do prefix_sum - Count pos within a substring
	for (i=0; i<Q; i++) 
		my_inclusive_scan_by_key <<<N/TPB, TPB>>> ((pos+1)+i*N, N);

	if  (R>0) 
		my_inclusive_scan_by_key <<<N/TPB, TPB>>> ((pos+1)+N*Q, R);
	
#if 0
	char *pchar = (char *)calloc (sizeof(char) , text_size);
	cudaMemcpy (pchar, text, sizeof(char) * text_size, cudaMemcpyDeviceToHost);

	int  *pint = (int *)calloc (sizeof(int), text_size);
	cudaMemcpy (pint, pos, sizeof(int)*text_size, cudaMemcpyDeviceToHost);

	for (i=0; i< text_size; i++) printf ("%c:%d ", pchar[i], pint[i]); 
#endif
	
}
