#include "LightTracker.h"
namespace bs
{
	LightTracker::LightTracker(bs::VideoCaptureYUV* video)
		: m_video(video)
	{
		//initialize windows
		cv::namedWindow(m_sFrame, cv::WINDOW_KEEPRATIO);
		cv::resizeWindow(m_sFrame, m_video->getResolution() / 2);

		cv::namedWindow(m_sLightThresh1, cv::WINDOW_KEEPRATIO);
		cv::resizeWindow(m_sLightThresh1, m_video->getResolution() / 3);
		cv::namedWindow(m_sLightThresh2, cv::WINDOW_KEEPRATIO);
		cv::resizeWindow(m_sLightThresh2, m_video->getResolution() / 3);

		cv::namedWindow(m_sLightMask, cv::WINDOW_KEEPRATIO);
		cv::resizeWindow(m_sLightMask, m_video->getResolution() / 3);

		cv::namedWindow(m_sBulb, cv::WINDOW_KEEPRATIO);
		cv::resizeWindow(m_sBulb, m_video->getResolution() / 2);

	}

	int32_t LightTracker::start()
	{
		//thresh1
		m_video->read(m_imgFrame);
		m_frameCounter++;

		thresholdLights(m_imgFrame, m_imgLightThresh1);
		cv::imshow(m_sLightThresh1, m_imgLightThresh1);
		//thresh2
		m_video->read(m_imgFrame);
		m_frameCounter++;

		thresholdLights(m_imgFrame, m_imgLightThresh2);
		cv::imshow(m_sLightThresh2, m_imgLightThresh2);

		createLightMask(m_imgLightThresh1, m_imgLightThresh2, m_imgLightMask);
		cv::imshow(m_sLightMask, m_imgLightMask);

		while (true)
		{

			if (cv::waitKey(5) == 27) break;
			if (m_video->read(m_imgFrame))
			{
				//cv::waitKey(0);
				m_bulbPos = detectLight(m_imgFrame, m_imgLightMask);
				cv::circle(m_imgFrame, m_bulbPos, 10, cv::Scalar(0, 255, 0), 3);
				cv::imshow(m_sFrame, m_imgFrame);

				std::cout << m_frameCounter << m_bulbPos << std::endl;

				m_frameCounter++;
			}
			else return bs::END_OF_FILE; //TODO - handle state in main
		}
	}

	LightTracker::~LightTracker()
	{
		
	}

	void LightTracker::thresholdLights(const cv::Mat& frame, cv::Mat& imgThresh)
	{
		cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_DILATE, cv::Size(9, 9));
		//cv::Mat kernel_erode = cv::getStructuringElement()

		const int32_t thresh = 210;
		cv::cvtColor(frame, imgThresh, cv::COLOR_BGR2GRAY);
		cv::blur(imgThresh, imgThresh, cv::Size(9, 9));
		cv::dilate(imgThresh, imgThresh, kernel_dilate);
		cv::threshold(imgThresh, imgThresh, thresh, 255, cv::THRESH_TOZERO);
	}

	void LightTracker::createLightMask(const cv::Mat& frame1, const cv::Mat& frame2, cv::Mat& mask)
	{
		cv::bitwise_and(frame1, frame2, mask);
	}

	cv::Point2f LightTracker::detectLight(const cv::Mat& frame, const cv::Mat& mask)
	{
		int32_t x = 0, y = 0;
		cv::Mat imgThresh;
		cv::Mat imgBulb;
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));

		thresholdLights(frame, imgThresh);
		cv::absdiff(imgThresh, mask, imgBulb);
		cv::GaussianBlur(imgBulb, imgBulb, cv::Size(3, 3), 0);
		cv::threshold(imgBulb, imgBulb, 220, 255, cv::THRESH_BINARY);
		cv::dilate(imgBulb, imgBulb, kernel);

		cv::imshow(m_sBulb, imgBulb);

		//moments
		cv::Mat imgCanny;
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;

		cv::Canny(imgBulb, imgCanny, 50, 150, 3);
		cv::findContours(imgCanny, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

		std::vector<cv::Moments> mu(contours.size());

		for (int i = 0; i < contours.size(); i++)
		{
			mu[i] = cv::moments(contours[i], false);
		}

		std::vector<cv::Point2f> mc(contours.size());

		for (int i = 0; i < contours.size(); i++)
		{
			mc[i] = cv::Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
		}

		cv::Mat drawing(imgCanny.size(), CV_8UC3, cv::Scalar(255, 255, 255));

		for (int i = 0; i < contours.size(); i++)
		{
			cv::Scalar color = cv::Scalar(167, 151, 0);
			cv::drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
			cv::circle(drawing, mc[i], 4, -1, 0);


		}

		cv::imshow("drawing", drawing);

		//return cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
	}
}