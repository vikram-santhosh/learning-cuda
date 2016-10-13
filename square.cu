// compute the square of first 64 whole numbers using 64 threads on the device

#include <stdio.h>

__global__ void square(float *d_out,float *d_in)
{
	int idx = threadIdx.x;
	float f = (float) d_in[idx];
	d_out[idx] = f*f;
}

int main(int argc,char* argv[])
{
	const int ARRAY_SIZE = 64;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	//declaration 
	float h_in[ARRAY_SIZE] ,h_out[ARRAY_SIZE]; //host
	float *d_out ,*d_in; 	// device

	//generate the input
	for(int i=0;i<ARRAY_SIZE;i++)
		h_in[i] = i;

	//allocate memory on the device
	cudaMalloc((void**) &d_in , ARRAY_BYTES);
	cudaMalloc((void**) &d_out , ARRAY_BYTES);

	//tranfer data host to device
	cudaMemcpy(d_in,h_in,ARRAY_BYTES,cudaMemcpyHostToDevice);

	//launch kernel
	square<<<1,64>>>(d_out,d_in);

	//tranfer data form device to host
	cudaMemcpy(h_out,d_out,ARRAY_BYTES,cudaMemcpyDeviceToHost);

	//display results

	for(int i=0;i<ARRAY_SIZE;i++)
	{
		printf("%f", h_out[i]);
		(i%4 == 0) ? printf("\n") : printf("\t");
	}

	printf("\n");

}