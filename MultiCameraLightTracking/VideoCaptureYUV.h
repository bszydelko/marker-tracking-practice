#pragma once
#include <fstream>
#include <opencv2/opencv.hpp>

namespace bs
{
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

	};

	class VideoCaptureYUV
	{
	private:
		bool m_error{ false };
		std::ifstream m_file;
		std::string  m_filename;

		int32_t m_width;
		int32_t m_height;
		int32_t m_chromaSubsampling;

		FrameYUV* m_frame{ nullptr };
		int32_t m_numFrames{ 0 };

		int32_t m_frameID{ -1 };

	public:

		VideoCaptureYUV(const std::string& filename, int32_t width, int32_t height, int32_t chromaSubsampling);
		~VideoCaptureYUV();

		bool isOpened() { return !m_error; }
		bool read(cv::Mat& dst);
		bool read(cv::Mat& dst, int32_t frameNumber);
		int32_t getFrameID() const;
		int32_t getNumFrames() const;
		cv::Size getResolution() const;
	};

}