#pragma once
#include <fstream>
#include <opencv2/opencv.hpp>

struct FrameYUV
{
	uint8_t* m_Ypixels = nullptr;
	uint8_t* m_Upixels = nullptr;
	uint8_t* m_Vpixels = nullptr;

	int32_t m_width;
	int32_t m_height;
	int32_t m_chromaSubsampling;

	std::streamsize m_Ysize;
	std::streamsize m_Usize;
	std::streamsize m_Vsize;

	int32_t m_a;
	int32_t m_b;
	int32_t m_c;

	FrameYUV();
	FrameYUV(int32_t width, int32_t height, int32_t chromaSubsampling);
	~FrameYUV();

	cv::Mat cvtMat();
	

};

class VideoYUV
{
private:
	bool m_error;
	std::ifstream m_file;
	std::string  m_filename;

	int32_t m_width;
	int32_t m_height;
	int32_t m_chromaSubsampling;

	FrameYUV* m_frame;



public:

	VideoYUV(std::string filename, int32_t width, int32_t height, int32_t chromaSubsampling);
	~VideoYUV();
	bool isOpened() { return !m_error;  }
	bool read(cv::Mat &dst);

};

