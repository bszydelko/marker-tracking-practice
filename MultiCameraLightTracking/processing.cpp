#include "processing.h"

namespace bs
{
	void thresholdLights(const cv::Mat& imgFrame, cv::Mat& imgThresh)
	{
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
		const int32_t thresh = 210;
		cv::cvtColor(imgFrame, imgThresh, cv::COLOR_BGR2GRAY);
		cv::GaussianBlur(imgThresh, imgThresh, cv::Size(11, 11), 0);
		cv::dilate(imgThresh, imgThresh, kernel);
		cv::threshold(imgThresh, imgThresh, thresh, 255, cv::THRESH_TOZERO);


	}

	void createLightMask(const cv::Mat& frame1, const cv::Mat& frame2, cv::Mat& mask)
	{
		cv::bitwise_and(frame1, frame2, mask);
	}

	cv::Point2f detectLight(const cv::Mat& frame, const cv::Mat& mask)
	{
		int32_t x = 0, y = 0;
		cv::Mat imgThresh;
		cv::Mat imgBulb;
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));

		bs::thresholdLights(frame, imgThresh);
		cv::absdiff(imgThresh, mask, imgBulb);
		cv::GaussianBlur(imgBulb, imgBulb, cv::Size(3, 3), 0);
		cv::threshold(imgBulb, imgBulb, 215, 255, cv::THRESH_BINARY);
		cv::dilate(imgBulb, imgBulb, kernel);


		cv::Moments m = cv::moments(imgBulb, true);

		cv::imshow("bulb", imgBulb);

		return cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);


	}
}