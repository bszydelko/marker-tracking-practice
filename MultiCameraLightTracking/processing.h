#pragma once
#include "opencv2/opencv.hpp"

namespace bs
{
	void thresholdLights(const cv::Mat& frame, cv::Mat& mask);
	void createLightMask(const cv::Mat& frame1, const cv::Mat& frame2, cv::Mat& mask);
}