#ifndef BINS_H
#define BINS_H

#include <vector>

class Bins
{
public:
	struct BinEntry
	{
		int  numBinCircles;

		float* center;		//deviceMem
		float* color;			//deviceMem
		float* radius;			//deviceMem
	};

	Bins(int width) : 
		m_width(width),
		m_numCirSum(0),
		m_maxNumBinCir(0),
		m_binCnt(width * width)
	{
		m_bins.resize(width * width);
	}
	~Bins();

	void	addBin(	int ix, int iy, 
					float* devCen, float* devCol, float* devRad, 
					unsigned int* devPredicate, unsigned int* devPos, int numCirs);

	void	getDevSegmentedOffset	(unsigned int*& devSegOffset);
	/*Get segmented center color radius
	*/
	void	getDevSegCenColRad		(float*& devSegCen, float*& devSegCol, float*& devSegRad);

	unsigned int getNumBinCirs() const
	{
		return m_numCirSum;
	}

	unsigned int getMaxBinCirs() const
	{
		return m_maxNumBinCir;
	}

private:
	unsigned int			m_width;
	unsigned int			m_numCirSum; //
	unsigned int			m_maxNumBinCir;
	unsigned int			m_binCnt;
	std::vector<BinEntry>	m_bins;

	unsigned int*				m_devSegOffset;
	std::vector<unsigned int>	m_hstSegOffset; //used to form the segmented center color radius

	float*	m_devSegCenter;
	float*	m_devSegColor;
	float*	m_devSegRadius;
};


#endif