#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include "VideoCaptureYUV.h"

namespace bs
{
	class LightTracker
	{
	private:

		//window names
		std::string m_sFrame		{ "frame" };
		std::string m_sLightThresh1	{ "light thresh 1" };
		std::string m_sLightThresh2	{ "light htresh 2" };
		std::string m_sLightMask	{ "light mask" };
		std::string m_sBulb			{ "bulb" };

		//frames
		cv::Mat m_imgFrame;
		cv::Mat m_imgLightThresh1;
		cv::Mat m_imgLightThresh2;
		cv::Mat m_imgLightMask;

		bs::VideoCaptureYUV* m_video;
		uint32_t m_frameCounter{ 0 };

		cv::Point m_bulbPos;

	public:

		LightTracker(bs::VideoCaptureYUV* video);
		int32_t start();
		~LightTracker();
	protected:
		void thresholdLights(const cv::Mat& frame, cv::Mat& imgThresh);
		void createLightMask(const cv::Mat& frame1, const cv::Mat& frame2, cv::Mat& mask);
		cv::Point2f detectLight(const cv::Mat& frame, const cv::Mat& mask);

	};


	enum STATE
	{
		END_OF_FILE
	};

}