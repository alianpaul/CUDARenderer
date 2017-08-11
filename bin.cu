#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "bin.h"

//void prefixSum(unsigned int* input, unsigned int* output, int cnt);
__global__ void kernelCompact(float* devSrc, float* devDst,
							  unsigned int* devPredicate, unsigned int* devPos,
							  int numCirs, int offset);

Bins::~Bins()
{
	for (size_t i = 0; i < m_bins.size(); ++i)
	{
		if (m_bins[i].numBinCircles)
		{
			cudaFree(m_bins[i].center);
			cudaFree(m_bins[i].color);
			cudaFree(m_bins[i].radius);
		}
	}

	cudaFree(m_devSegOffset);
	cudaFree(m_devSegCenter);
	cudaFree(m_devSegColor);
	cudaFree(m_devSegRadius);
}

void
Bins::addBin(int ix, int iy,
			float* devCen, float* devCol, float* devRad,
			unsigned int* devPred, unsigned int* devPos, int numCirs)
{
	int offset = iy * m_width + ix;
	_ASSERT(offset < m_bins.size());

	//Get the size of the bin container
	unsigned int lastPos;
	cudaMemcpy(&lastPos, devPos + numCirs - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int lastPred;
	cudaMemcpy(&lastPred, devPred + numCirs - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	unsigned numBinCirs = (lastPred) ? lastPos + 1 : lastPos;
	m_numCirSum += numBinCirs;
	m_maxNumBinCir = std::max(numBinCirs, m_maxNumBinCir);

	//Initialize bin
	BinEntry& binE = m_bins[offset];
	binE.numBinCircles = numBinCirs;
	cudaMalloc(&binE.center,	sizeof(float) * 3 * numBinCirs);
	cudaMalloc(&binE.color,		sizeof(float) * 3 * numBinCirs);
	cudaMalloc(&binE.radius,	sizeof(float) * numBinCirs);

	//Compact the original data to bin
	dim3 blockDim(256);
	dim3 gridDim( (numCirs + blockDim.x - 1) / blockDim.x);
	//Compact the circle center
	kernelCompact<<<gridDim, blockDim>>>(devCen, binE.center, devPred, devPos, numCirs, 3);
	//Compact the color
	kernelCompact<<<gridDim, blockDim>>>(devCol, binE.color,  devPred, devPos, numCirs, 3);
	//Compact the radius
	kernelCompact<<<gridDim, blockDim>>>(devRad, binE.radius, devPred, devPos, numCirs, 1);
}

void
Bins::getDevSegmentedOffset(unsigned int*& devSegOffset)
{
	/*Prepare data on the host*/
	m_hstSegOffset.resize(m_binCnt, 0);
	for (size_t i = 1; i < m_binCnt; ++i)
	{
		m_hstSegOffset[i] = m_bins[i - 1].numBinCircles + m_hstSegOffset[i - 1];
	}

	cudaMalloc(&m_devSegOffset, sizeof(unsigned int) * m_binCnt);
	cudaMemcpy(m_devSegOffset, &m_hstSegOffset[0], sizeof(unsigned int) * m_binCnt, cudaMemcpyHostToDevice);
	
	devSegOffset = m_devSegOffset;
}


void
Bins::getDevSegCenColRad(float* &devSegCen, float* &devSegCol, float* &devSegRad)
{
	cudaMalloc(&m_devSegCenter, sizeof(float) * 3 * m_numCirSum);
	cudaMalloc(&m_devSegColor,	sizeof(float) * 3 * m_numCirSum);
	cudaMalloc(&m_devSegRadius, sizeof(float) *		m_numCirSum);

	for (size_t i = 0; i < m_binCnt; ++i)
	{
		unsigned offset  = m_hstSegOffset[i];
		unsigned offset3 = offset * 3;

		//copy center
		cudaMemcpy(	m_devSegCenter + offset3, m_bins[i].center,
					sizeof(float) * 3 * m_bins[i].numBinCircles,
					cudaMemcpyDeviceToDevice);

		//copy color
		cudaMemcpy( m_devSegColor + offset3, m_bins[i].color,
					sizeof(float) * 3 * m_bins[i].numBinCircles,
					cudaMemcpyDeviceToDevice);

		//copy radius
		cudaMemcpy( m_devSegRadius + offset, m_bins[i].radius,
					sizeof(float) * m_bins[i].numBinCircles,
					cudaMemcpyDeviceToDevice);
	}

	devSegCen = m_devSegCenter;
	devSegCol = m_devSegColor;
	devSegRad = m_devSegRadius;
}


