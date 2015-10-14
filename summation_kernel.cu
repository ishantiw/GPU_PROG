/*
Author: Ishan Tiwari & Shashi Mohan Reddy Ravula
Description: The kernel file which has the functions that will run on GPU
*/

// GPU kernel


//This is version1 and strategy 2 as explained in the report

__global__ void summation_kernel1(int data_size, float * data_out)
{
	// TODO
	float sum = 0;
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int nthreads = blockDim.x * gridDim.x;

	data_out[tid] = 0;
	for (int i = tid; i<data_size; i = i + nthreads)
	{
		if (i % 2 == 0)
		{
			sum = sum + 1 / float(i + 1);
		}
		else if (i % 2 == 1)
		{
			sum = sum - 1 / float(i + 1);
		}
	}
	data_out[tid] = sum;
}

//This is version2 and strategy 1 as explained in the report
__global__ void summation_kernel1_version2(int data_size, float * data_out)
{
	// TODO
	float sum = 0;
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int nthreads = blockDim.x * gridDim.x;
	int begin = tid * nthreads;
	int end = begin + nthreads;
	data_out[tid] = 0;
	for (int i = begin; i<end; i = i++)
	{
		if (i % 2 == 0)
		{
			sum = sum + 1 / float(i + 1);
		}
		else if (i % 2 == 1)
		{
			sum = sum - 1 / float(i + 1);
		}
	}
	data_out[tid] = sum;
}

//Getting the partial results by returning total sum from each block 
//and also allocating size for the shared memory used for partial results
__global__ void summation_kernel2(int data_size, float * data_out, float *partial)
{
	// TODO
	extern __shared__ float sdata[];

	// each thread loads one element from global to shared mem

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i< data_size){
		sdata[tid] = data_out[i];
	}
	else{
		sdata[tid] = 0;
	}
	__syncthreads();
	// Reduction
	int flag;
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		flag = tid * (2 * s);
		if (flag < blockDim.x) {
			sdata[flag] += sdata[flag + s];
		}
		__syncthreads();
	}
	// Writing the result per block back to the global memory
	if (tid == 0) {
		partial[blockIdx.x] = sdata[0];
	}
}


//This kernel will take the partial result from "summation_kernel2" in "partial_gpu" variable
__global__ void summation_kernel3(float * data_in, float * data_out)
{
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	sdata[tid] = data_in[tid];

	__syncthreads();
	//Reduction
	for (unsigned int s = 1; s < blockDim.x; s *= 2)
	{
		int flag = 2 * s * tid;
		if (flag < blockDim.x)
		{
			sdata[flag] += sdata[flag + s];
		}
		__syncthreads();
	}
	// Writing the results back to the global memory
	if (tid == 0){
		data_out[0] = sdata[0];
	}
}