#include <iostream>
#include <cuda.h> 
#include <stdlib.h>
#include <time.h>

using namespace std;

__global__ void vector_add(int *d_vec1,int *d_vec2,int *d_vec3)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	d_vec3[idx] = d_vec1[idx] + d_vec2[idx]; 
}
int main()
{
	const int num_block = 1000;
	const int thread_in_block = 512;
	const int SIZE = num_block * thread_in_block;
	const int BYTES = SIZE * sizeof (int);

	clock_t start_t, end_t;
	double time_spent;

	int h_vec1[SIZE], h_vec2[SIZE],h_vec3[SIZE]; // c = a + b
	int *d_vec1, *d_vec2, *d_vec3;


	for(int i=0;i<SIZE;i++)
	{
		h_vec1[i] = rand()%100;
		h_vec2[i] = rand()%100;
		h_vec3[i] = 0;
	}

	cudaMalloc((void**) &d_vec1, BYTES);
	cudaMalloc((void**) &d_vec2, BYTES);
	cudaMalloc((void**) &d_vec3, BYTES);

	cudaMemcpy(d_vec1,h_vec1,BYTES,cudaMemcpyHostToDevice);
	cudaMemcpy(d_vec2,h_vec2,BYTES,cudaMemcpyHostToDevice);

	start_t = clock();
	vector_add<<<num_block,thread_in_block>>>(d_vec1,d_vec2,d_vec3);
	end_t = clock();
	time_spent = (double) (end_t-start_t)/CLOCKS_PER_SEC;
	
	cudaMemcpy(h_vec3,d_vec3,BYTES,cudaMemcpyDeviceToHost);

	// for(int i=0;i<SIZE;i++)
	// {
	// 	cout<<h_vec3[i];
	// 	(i%5 == 0) ? cout<<endl : cout<<"\t";
	// }

	cout<<"Execution time = "<< time_spent<<endl;
}