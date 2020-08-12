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

	class CaptureYUV
	{
	private:
		bool m_error{ false };
		std::ifstream m_file;
		std::string  m_filename;
		std::streampos m_filePos{ 0 };

		int32_t m_width;
		int32_t m_height;
		int32_t m_chromaSubsampling;

		FrameYUV* m_frame{ nullptr };
		int32_t m_numFrames{ 0 };

		int32_t m_frameID;

		int32_t m_frameStep{ 0 };

	public:

		CaptureYUV(const std::string& filename, int32_t width, int32_t height, int32_t chromaSubsampling, const int32_t frameStep);
		~CaptureYUV();

		bool isOpened() { return !m_error; }
		bool read(cv::Mat& dst);
		int32_t getFrameID() const;
		int32_t getNumFrames() const;
		cv::Size getResolution() const;
		std::string getFilename() const;
		int32_t getFrameStep() const;
	};

}