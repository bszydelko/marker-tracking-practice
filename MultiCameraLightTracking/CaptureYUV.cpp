#include "CaptureYUV.h"

namespace bs
{
	FrameYUV::FrameYUV(int32_t width, int32_t height, int32_t chromaSubsampling)
	{
		m_width = width;
		m_height = height;
		m_chromaSubsampling = chromaSubsampling;

		m_c = m_chromaSubsampling % 10;
		m_b = (m_chromaSubsampling / 10) % 10;
		m_a = (m_chromaSubsampling / 100) % 10;

		m_Ysize = m_width * m_height;
		m_Usize = (m_width / m_b) * (m_height / m_b);
		m_Vsize = (m_width / m_b) * (m_height / m_b);


		this->m_Ypixels = new uint8_t[m_Ysize];
		this->m_Upixels = new uint8_t[m_Usize];
		this->m_Vpixels = new uint8_t[m_Vsize];
	}

	FrameYUV::~FrameYUV()
	{
		delete[] m_Ypixels;
		delete[] m_Upixels;
		delete[] m_Vpixels;
	}
	//=================================================================
	CaptureYUV::CaptureYUV(const std::string& filename, int32_t width, int32_t height, int32_t chromaSubsampling, const int32_t frameStep)
		: m_filename(filename), m_width(width), m_height(height), m_chromaSubsampling(chromaSubsampling), m_frameStep(frameStep)
	{
		m_file.open(m_filename, std::ios::binary);

		if (!m_file.good())
		{
			m_error = true;
			return;
		}
		
		m_frame = new FrameYUV(m_width, m_height, m_chromaSubsampling);
		//calculate frame number
		m_file.seekg(0, std::ios::end);
		m_numFrames = m_file.tellg() / (m_frame->m_Ysize + m_frame->m_Usize + m_frame->m_Vsize);
		m_file.seekg(0, std::ios::beg);

		//set pointer
		if (m_frameStep == -1) {
			m_frameID = m_numFrames;
			m_filePos = (m_frameID - 1) * (m_frame->m_Ysize + m_frame->m_Usize + m_frame->m_Vsize);
			m_file.seekg(m_filePos, std::ios::beg);
		}
		else {
			m_frameID = -1;
			m_filePos = (m_frameID + 1) * (m_frame->m_Ysize + m_frame->m_Usize + m_frame->m_Vsize);
			m_file.seekg(m_filePos, std::ios::beg);
		}
		
	}

	CaptureYUV::~CaptureYUV()
	{
		m_file.close();
		delete m_frame;
	}

	bool CaptureYUV::read(cv::Mat& dst)
	{
		
		if (m_file.eof())
			return false; 

		m_frameID += m_frameStep;
		m_filePos =  m_frameID * (m_frame->m_Ysize + m_frame->m_Usize + m_frame->m_Vsize);
		m_file.seekg(m_filePos, std::ios::beg);

		m_file.read((char*)m_frame->m_Ypixels, m_frame->m_Ysize);
		m_file.read((char*)m_frame->m_Upixels, m_frame->m_Usize);
		m_file.read((char*)m_frame->m_Vpixels, m_frame->m_Vsize);

		
		dst = cv::Mat::zeros(cv::Size(m_width, m_height), CV_8UC3);
		float B = 0, G = 0, R = 0;
		uint8_t Y = 0, U = 0, V = 0;
		int32_t b_idx = 0, g_idx = 1, r_idx = 2; //indices


		for (size_t r = 0; r < m_height; r++)
		{
			for (size_t c = 0; c < m_width; c++)
			{
				Y = m_frame->m_Ypixels[r * m_width + c];
				U = m_frame->m_Upixels[(r / m_frame->m_b) * (m_width / m_frame->m_b) + (c / m_frame->m_b)];
				V = m_frame->m_Vpixels[(r / m_frame->m_b) * (m_width / m_frame->m_b) + (c / m_frame->m_b)];

				/*R = Y + 1.4075f * (V - 128);
				G = Y - 0.3455f * (U - 128) - (0.7169 * (V - 128));
				B = Y + 1.7790f * (U - 128);

				if (R < 0) { R = 0; } if (G < 0) { G = 0; } if (B < 0) { B = 0; }
				if (R > 255) { R = 255; } if (G > 255) { G = 255; } if (B > 255) { B = 255; }*/

				dst.at<cv::Vec3b>(r, c)[b_idx] = Y;
				dst.at<cv::Vec3b>(r, c)[g_idx] = U;
				dst.at<cv::Vec3b>(r, c)[r_idx] = V;
			}
		}

		cv::cvtColor(dst, dst, cv::COLOR_YUV2BGR);
		

		return true;
	}

	int32_t CaptureYUV::getFrameID() const
	{
		return int32_t(m_frameID);
	}

	int32_t CaptureYUV::getNumFrames() const
	{
		return int32_t(m_numFrames);
	}

	cv::Size CaptureYUV::getResolution() const
	{
		return cv::Size(m_width, m_height);
	}

	std::string CaptureYUV::getFilename() const
	{
		return std::string(m_filename);
	}

	int32_t CaptureYUV::getFrameStep() const
	{
		return int32_t(m_frameStep);
	}

}