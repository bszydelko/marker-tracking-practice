#include "processing.h"

namespace bs
{
	void thresholdLights(const cv::Mat& frame, cv::Mat& mask)
	{
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
		const int32_t thresh = 210;
		cv::cvtColor(frame, mask, cv::COLOR_BGR2GRAY);
		cv::GaussianBlur(mask, mask, cv::Size(11, 11), 0);
		cv::dilate(mask, mask, kernel);
		cv::threshold(mask, mask, thresh, 255, cv::THRESH_TOZERO);


	}

	void createLightMask(const cv::Mat& frame1, const cv::Mat& frame2, cv::Mat& mask)
	{
		cv::bitwise_and(frame1, frame2, mask);
	}
}