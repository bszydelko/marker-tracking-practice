#include "LightTracker.h"
namespace bs
{
	LightTracker::LightTracker(bs::VideoCaptureYUV* video)
		: m_video(video)
	{
		//initialize windows
		cv::namedWindow(m_sCurrentFrame, cv::WINDOW_KEEPRATIO);
		cv::resizeWindow(m_sCurrentFrame, m_video->getResolution() / 2);

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
		m_video->read(m_imgCurrentFrame);
		m_frameCounter++;

		thresholdLights(m_imgCurrentFrame, m_imgLightThresh1);
		cv::imshow(m_sLightThresh1, m_imgLightThresh1);
		//thresh2
		m_video->read(m_imgCurrentFrame);
		m_frameCounter++;

		thresholdLights(m_imgCurrentFrame, m_imgLightThresh2);
		cv::imshow(m_sLightThresh2, m_imgLightThresh2);

		createLightMask(m_imgLightThresh1, m_imgLightThresh2, m_imgLightMask);
		cv::imshow(m_sLightMask, m_imgLightMask);

		while (true)
		{

			if (cv::waitKey(5) == 27) break;
			if (m_video->read(m_imgCurrentFrame))
			{
				cv::waitKey(0);
				m_bulbPos = detectLight(m_imgCurrentFrame, m_imgLightMask);
				cv::circle(m_imgCurrentFrame, m_bulbPos, 10, cv::Scalar(0, 255, 0), 3);
				cv::imshow(m_sCurrentFrame, m_imgCurrentFrame);

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
		m_kernelDilate_thresholdLights = cv::getStructuringElement(cv::MORPH_DILATE, cv::Size(9, 9));
		//cv::Mat m_kernelErode_detectLight = cv::getStructuringElement()

		m_thresh_thresholdLights = 210;
		cv::cvtColor(frame, imgThresh, cv::COLOR_BGR2GRAY);
		cv::blur(imgThresh, imgThresh, cv::Size(9, 9));
		cv::dilate(imgThresh, imgThresh, m_kernelDilate_thresholdLights);
		cv::threshold(imgThresh, imgThresh, m_thresh_thresholdLights, 255, cv::THRESH_TOZERO);
	}

	void LightTracker::createLightMask(const cv::Mat& frame1, const cv::Mat& frame2, cv::Mat& mask)
	{
		cv::bitwise_and(frame1, frame2, mask);
	}

	cv::Point2f LightTracker::detectLight(const cv::Mat& frame, const cv::Mat& mask)
	{
		m_kernelDilate_detectLight = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));

		thresholdLights(frame, m_imgThresh_detectLight);
		cv::absdiff(m_imgThresh_detectLight, mask, m_imgBulb_detectLight);
		cv::GaussianBlur(m_imgBulb_detectLight, m_imgBulb_detectLight, cv::Size(3, 3), 0);
		cv::threshold(m_imgBulb_detectLight, m_imgBulb_detectLight, 220, 255, cv::THRESH_BINARY);
		cv::dilate(m_imgBulb_detectLight, m_imgBulb_detectLight, m_kernelDilate_detectLight);

		cv::imshow(m_sBulb, m_imgBulb_detectLight);
	
		//check if bulb overlaps with another light source
		//check if another object cover mask




		//contours
	
		cv::Canny(
			m_imgBulb_detectLight, 
			m_imgCanny_detectLight, 50, 150, 3);
		cv::findContours(
			m_imgCanny_detectLight, 
			m_vecContours_detectLight, 
			m_vecHierarchy_detectLight, 
			cv::RETR_TREE, 
			cv::CHAIN_APPROX_SIMPLE, 
			cv::Point(0, 0));

		m_vecMoments_detectLight.clear();
		m_vecMoments_detectLight.reserve(m_vecContours_detectLight.size());
		m_vecCentralMoments_detectLight.clear();
		m_vecCentralMoments_detectLight.reserve(m_vecContours_detectLight.size());
		

		for (int i = 0; i < m_vecContours_detectLight.size(); i++)
		{
			m_vecMoments_detectLight[i] = 
				cv::moments(m_vecContours_detectLight[i], false); 		//moments for each blob / contour
			m_vecCentralMoments_detectLight[i] = 
				cv::Point2f(											//center of each blob / contour
					m_vecMoments_detectLight[i].m10 / m_vecMoments_detectLight[i].m00, 
					m_vecMoments_detectLight[i].m01 / m_vecMoments_detectLight[i].m00); 		
		}

		cv::Mat drawing(m_imgCanny_detectLight.size(), CV_8UC3, cv::Scalar(255, 255, 255));
		cv::Scalar color = cv::Scalar(167, 151, 0);

		for (int i = 0; i < m_vecContours_detectLight.size(); i++)
		{
			cv::drawContours(drawing, m_vecContours_detectLight, i, color, 2, 8, m_vecHierarchy_detectLight, 0, cv::Point());
			cv::circle(drawing, m_vecCentralMoments_detectLight[i], 4, -1, 0);
		}

		cv::Point2f retPoint;

		if (m_vecContours_detectLight.size() == 1)
		{

		}

		cv::imshow("drawing", drawing);






		//return cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
		return cv::Point(0, 0);
	}
	
}