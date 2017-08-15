#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "helper.cuh"

__global__ void preScan(unsigned int* deviceInput, unsigned int* deviceOutput, int cnt,
						unsigned int* deviceSum)
{
	extern __shared__ unsigned int temp[];
	int cntInB = blockDim.x * 2;
	int idxInG = cntInB * blockIdx.x + threadIdx.x;

	int idxInB = threadIdx.x;
	temp[2 * idxInB]		= 0;
	temp[2 * idxInB +1]		= 0;

	if (idxInG < cnt)
	{
		temp[idxInB] = deviceInput[idxInG];
	}
	
	if (idxInG + blockDim.x < cnt)
	{
		temp[idxInB + blockDim.x] = deviceInput[idxInG + blockDim.x];
	}

	int offset = 1;
	for (int d = cntInB >> 1; d > 0; d>>=1)
	{
		__syncthreads();
		if (threadIdx.x < d)
		{
			int ai = offset - 1 + offset * (threadIdx.x * 2);
			int bi = ai + offset;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	__syncthreads();
	//before clear the last element, move the last element to deviceSums.
	if (threadIdx.x == 0)
	{
		deviceSum[blockIdx.x] = temp[cntInB - 1];
		temp[cntInB - 1] = 0;
	}
	
	//downsweep
	for (int d = 1; d < cntInB; d *=2)
	{
		offset >>= 1;
		__syncthreads();

		if (threadIdx.x < d)
		{

			int ai = offset - 1 + offset * (threadIdx.x * 2);
			int bi = ai + offset;
			unsigned int be = temp[bi];
			temp[bi] += temp[ai];
			temp[ai] = be;
		}
	}

	if (idxInG < cnt)
	{
		deviceOutput[idxInG] = temp[idxInB];
	}

	if (idxInG + blockDim.x < cnt)
	{
		deviceOutput[idxInG + blockDim.x] = temp[idxInB + blockDim.x];
	}
}

__global__ void addInc(unsigned int* deviceInput, unsigned int* deviceOutput, int eleCnt,
					   unsigned int* deviceInc)
{
	/*
	__shared__ int inc;
	if (threadIdx.x == 0)
	{
		inc = deviceInc[blockIdx.x];
	}
	__syncthreads();
	*/
	int inc = deviceInc[blockIdx.x];
	
	int cntInB = blockDim.x * 2;
	int idxInG = blockIdx.x * cntInB + threadIdx.x;

	if (idxInG < eleCnt)
	{
		deviceOutput[idxInG] = deviceInput[idxInG] + inc;
	}

	if (idxInG + blockDim.x < eleCnt)
	{
		deviceOutput[idxInG + blockDim.x] = deviceInput[idxInG + blockDim.x] + inc;
	}

}


/*input:	allocated and initialized device memory
* output:	allocated device memory
* cnt:		size
*
* return the sum of all element;
*/
unsigned int prefixSum(unsigned int* deviceInput, unsigned int* deviceOutput, int eleCnt)
{

	/*Test:	
	int eleCnt = 1025;
	unsigned int* deviceInput;
	cudaMalloc(&deviceInput, sizeof(unsigned int) * eleCnt);
	unsigned int* deviceOutput;
	cudaMalloc(&deviceOutput, sizeof(unsigned int) * eleCnt);

	unsigned int* hostInput;
	hostInput = (unsigned int*)malloc(sizeof(unsigned int) * eleCnt);
	for (size_t i = 0; i < eleCnt; ++i)
	{
		hostInput[i] = 1;
	}
	cudaMemcpy(deviceInput, hostInput, sizeof(unsigned int) * eleCnt, cudaMemcpyHostToDevice);
	*/

	dim3 blockDim(256);
	int	 eleCntInB = blockDim.x * 2;
	unsigned int sharedMemSize = eleCntInB * sizeof(unsigned int);

	dim3 gridDim((eleCnt+ eleCntInB - 1) / eleCntInB);
	int blockCnt = gridDim.x;

	unsigned int* deviceSum;
	cudaMalloc(&deviceSum, sizeof(unsigned int)*blockCnt);
	unsigned int* deviceInc;
	cudaMalloc(&deviceInc, sizeof(unsigned int)*blockCnt);
	unsigned int* deviceTotalSum;
	cudaMalloc(&deviceTotalSum, sizeof(unsigned int));

	preScan<<<gridDim, blockDim, sharedMemSize>>>(deviceInput, deviceOutput, eleCnt,
												  deviceSum);
	preScan<<<1, blockDim, sharedMemSize>>>(deviceSum, deviceInc, blockCnt,
											deviceTotalSum);
	addInc<<<gridDim, blockDim>>>(deviceOutput, deviceOutput, eleCnt,
								  deviceInc);
	
	/*Test Output:
	unsigned int* hostScanOut = (unsigned int*)malloc(sizeof(unsigned int) * eleCnt);
	cudaMemcpy(hostScanOut, deviceOutput, sizeof(unsigned int) * eleCnt, cudaMemcpyDeviceToHost);
	printf("Final result\n");
	for (size_t i = 0; i < eleCnt; ++i)
	{
		printf("%d ", hostScanOut[i]);
	}
	*/

	unsigned int hostTotalSum; 
	cudaMemcpy(&hostTotalSum, deviceTotalSum, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	cudaFree(deviceInc);
	cudaFree(deviceSum);
	cudaFree(deviceTotalSum);

	return hostTotalSum;
}