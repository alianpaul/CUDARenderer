#include <cuda.h>
#include <cuda_runtime.h>

/*numCirs: num of total circles
*/
__global__ void kernelCompact(float* devSrc, float* devDst,
							  unsigned int* devPredicate, unsigned int* devPos,
							  int numCirs, int offset)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; //index of the circles

	if (idx >= numCirs)
	{
		return;
	}

	unsigned int isIn = devPredicate[idx];
	if (isIn != 1)
	{
		return;
	}

	unsigned int pos = devPos[idx];

	idx *= offset;
	pos *= offset;
	
	for (int i = 0; i < offset; ++i)
	{
		devDst[pos + i] = devSrc[idx + i];
	}
}