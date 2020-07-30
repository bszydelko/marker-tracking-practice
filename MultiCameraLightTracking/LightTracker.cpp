#include "LightTracker.h"
namespace bs
{
	LightTracker::LightTracker(bs::VideoCaptureYUV* video)
		: m_video(video)
	{
		//initialize windows
		cv::namedWindow(m_sPreviousFrame, cv::WINDOW_KEEPRATIO);
		cv::resizeWindow(m_sPreviousFrame, m_video->getResolution() / 2);

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

		cv::namedWindow(m_sContoursMoments, cv::WINDOW_KEEPRATIO);
		cv::resizeWindow(m_sContoursMoments, m_video->getResolution());

		m_kernelDilate_thresholdLights = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
		m_kernelDilate_detectBulb = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));

	}

	int32_t LightTracker::start()
	{
		//thresh1
		m_video->read(m_imgCurrentFrame);
		m_frameCounter++;
		thresholdLights(m_imgCurrentFrame, m_imgLightThresh1);

		//thresh2
		m_video->read(m_imgCurrentFrame);
		m_frameCounter++;
		thresholdLights(m_imgCurrentFrame, m_imgLightThresh2);

		createLightMask(m_imgLightThresh1, m_imgLightThresh2, m_imgLightMask);

		m_video->read(m_imgFirstBulbFrame);
		m_frameCounter++;
		cv::Point2d bulbFirstPos = detectBulbInFirstFrame(m_imgFirstBulbFrame, m_imgLightMask);
		m_vecBulbs.emplace_back(bs::Bulb(bulbFirstPos, m_frameCounter));

		cv::circle(m_imgFirstBulbFrame, bulbFirstPos, 10, cv::Scalar(234, 247, 29), 3);
		cv::imshow("first bulb pos", m_imgFirstBulbFrame);
		m_imgFirstBulbFrame.copyTo(m_imgPreviousFrame);
	
		//imshow triggers
		bool previousFrame = 0;
		bool currentFrame = 1;
		bool lightThresh1 = 0;
		bool lightThresh2 = 0;
		bool lightMask = 0;
		bool bulb = 1;
		bool contoursMoments = 1;

		while (m_video->read(m_imgCurrentFrame))
		{
			m_frameCounter++;

			if (cv::waitKey(5) == 27) break;
			

			m_bulbPos = detectBulb(m_imgCurrentFrame, m_imgLightMask);
			cv::circle(m_imgCurrentFrame, m_bulbPos, 10, cv::Scalar(0, 255, 0), 3);

			
			imshow(previousFrame, currentFrame, lightThresh1, lightThresh2, lightMask, bulb, contoursMoments);
			std::cout << m_frameCounter << m_bulbPos << std::endl;

			m_imgCurrentFrame.copyTo(m_imgPreviousFrame);
			//cv::waitKey(0);

		}

		return 1;
	}

	LightTracker::~LightTracker()
	{

	}

	void LightTracker::thresholdLights(const cv::Mat& frame, cv::Mat& imgThresh)
	{
		//cv::Mat m_kernelErode_detectLight = cv::getStructuringElement()

		cv::cvtColor(frame, imgThresh, cv::COLOR_BGR2GRAY);
		cv::blur(imgThresh, imgThresh, cv::Size(9, 9));
		cv::dilate(imgThresh, imgThresh, m_kernelDilate_thresholdLights);
		cv::threshold(imgThresh, imgThresh, m_thresh_thresholdLights, 255, cv::THRESH_TOZERO);
	}

	void LightTracker::createLightMask(const cv::Mat& frame1, const cv::Mat& frame2, cv::Mat& mask)
	{
		cv::bitwise_and(frame1, frame2, mask);
	}

	cv::Point2d LightTracker::detectBulb(const cv::Mat& frame, const cv::Mat& mask)
	{
		thresholdLights	(frame, m_imgThresh_detectBulb);
		cv::absdiff		(m_imgThresh_detectBulb, mask, m_imgBulb_detectBulb);
		cv::GaussianBlur(m_imgBulb_detectBulb, m_imgBulb_detectBulb, cv::Size(3, 3), 0);
		cv::threshold	(m_imgBulb_detectBulb, m_imgBulb_detectBulb, 220, 255, cv::THRESH_BINARY);
		cv::dilate		(m_imgBulb_detectBulb, m_imgBulb_detectBulb, m_kernelDilate_detectBulb);

		//check if another object cover mask
		cv::Mat kernel_e = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
		cv::Mat kernel_d = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

		cv::Mat bulb;
		m_imgBulb_detectBulb.copyTo(bulb);
		cv::erode(bulb, bulb, kernel_e);
		bool retVal = false;
		for (size_t r = 0; r < frame.rows; r++)
		{
			for (size_t c = 0; c < frame.cols; c++)
			{
				if (mask.at<uint8_t>(r, c) && m_imgBulb_detectBulb.at<uint8_t>(r, c))
				{
					retVal = true;
					bulb.at<uint8_t>(r, c) = 0;
				}
			}
		}

		cv::dilate(bulb, bulb, kernel_d);
		bulb.copyTo(m_imgBulb_detectBulb);
		//check if another object cover mask

		//contours
		cv::Canny(
			m_imgBulb_detectBulb, 
			m_imgCanny_detectBulb, 50, 150, 3);
		cv::findContours(
			m_imgCanny_detectBulb, 
			m_vecContours_detectBulb, 
			m_vecHierarchy_detectBulb, 
			cv::RETR_EXTERNAL, 
			cv::CHAIN_APPROX_NONE, 
			cv::Point(0, 0));

		m_vecMoments_detectBulb.clear();
		m_vecMoments_detectBulb.reserve(m_vecContours_detectBulb.size());
		m_vecCentralMoments_detectBulb.clear();
		m_vecCentralMoments_detectBulb.reserve(m_vecContours_detectBulb.size());
		//contours

		for (int i = 0; i < m_vecContours_detectBulb.size(); i++)
		{
			m_vecMoments_detectBulb[i] = 
				cv::moments(m_vecContours_detectBulb[i], false); 		//moments for each blob / contour
			m_vecCentralMoments_detectBulb[i] = 
				cv::Point2d(											//center of each blob / contour
					m_vecMoments_detectBulb[i].m10 / m_vecMoments_detectBulb[i].m00, 
					m_vecMoments_detectBulb[i].m01 / m_vecMoments_detectBulb[i].m00); 		
		}

		bs::Bulb prevBulb = m_vecBulbs.back();
		std::cout << prevBulb;
		cv::Point2d retPoint;

		cv::Point2d predictedPosition = predictAverage();
		cv::drawMarker(m_imgCurrentFrame, predictedPosition, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 20, 3);

		

		if (m_vecContours_detectBulb.size() == 0)
		{
			retPoint = cv::Point(0, 0);
			m_vecBulbs.emplace_back(bs::Bulb(predictedPosition, m_frameCounter, false));
		}
		else
		{
			//find blob by predicted point
			double area1 = 0.0;
			double area2 = 0.0;
			int idx = 0;

			area1 = cv::contourArea(m_vecContours_detectBulb[0]);
			retPoint = m_vecCentralMoments_detectBulb[0];

			for (int i = 1; i < m_vecContours_detectBulb.size(); i++) {
				area2 = cv::contourArea(m_vecContours_detectBulb[i]);
				if (area2 > area1) {
					retPoint = m_vecCentralMoments_detectBulb[i];
					area1 = area2;
					idx = i;
				}
			}

			bool bulbOnMask = bulbVsMask(m_vecContours_detectBulb[idx], mask);
			std::cout << "bulbVSmask: " << bulbOnMask << std::endl;
			m_vecBulbs.emplace_back(bs::Bulb(retPoint, m_frameCounter));
			m_vecBulbs.back().setMotion(&prevBulb);

		}

		return retPoint;
	}

	cv::Point2d LightTracker::detectBulbInFirstFrame(const cv::Mat& frame, const cv::Mat& mask)
	{
		cv::Mat								frame_thresh;
		cv::Mat								frame_bulb;
		cv::Mat								frame_canny;
		std::vector<std::vector<cv::Point>> vecContours;
		std::vector<cv::Vec4i>				vecHierarchy;
		std::vector<cv::Moments>			vecMoments;
		std::vector<cv::Point2f>			vecCentralMoments;

		thresholdLights	(frame, frame_thresh);
		cv::absdiff		(frame_thresh, mask, frame_bulb);
		cv::GaussianBlur(frame_bulb, frame_bulb, cv::Size(3, 3), 0);
		cv::threshold	(frame_bulb, frame_bulb, 220, 255, cv::THRESH_BINARY);
		cv::dilate		(frame_bulb, frame_bulb, m_kernelDilate_detectBulb);

		cv::Canny(
			frame_bulb,
			frame_canny, 50, 150, 3);
		cv::findContours(
			frame_canny,
			vecContours,
			vecHierarchy,
			cv::RETR_TREE,
			cv::CHAIN_APPROX_SIMPLE,
			cv::Point(0, 0));

		vecMoments.reserve(vecContours.size());
		vecCentralMoments.reserve(vecContours.size());

		for (int i = 0; i < vecContours.size(); i++){
			vecMoments[i] =
				cv::moments(vecContours[i], false); 		//moments for each blob / contour
			vecCentralMoments[i] =
				cv::Point2f(											//center of each blob / contour
					vecMoments[i].m10 / vecMoments[i].m00,
					vecMoments[i].m01 / vecMoments[i].m00);
		}

		double area1 = 0.0;
		double area2 = 0.0;
		cv::Point2d retPoint;

		if (vecContours.size() > 1){
			area1 = cv::contourArea(vecContours[0]);
			retPoint = vecCentralMoments[0];

			for (int i = 1; i < vecContours.size(); i++){
				area2 = cv::contourArea(vecContours[i]);
				if (area2 > area1){
					retPoint = vecCentralMoments[i];
					area1 = area2;
				}
			}
		}
		return retPoint;
	}

	bool LightTracker::bulbVsMask(const std::vector<cv::Point>& bulbContour, const cv::Mat& mask)
	{
		for (const auto& bC : bulbContour)
		{
			if (mask.at<uint8_t>(bC) >= 1) 
				return true;
		}
		return false;
	}

	cv::Point2d LightTracker::predictAverage()
	{
		cv::Point2d predictedPosition;
		int frames = m_vecBulbs.size();
		
		if (frames == 0) return cv::Point2d(-1, -1);
		else if (frames == 1) return m_vecBulbs[0].m_position;
		else if (frames == 2)
		{
			double deltaX = m_vecBulbs[frames - 1].m_position.x - m_vecBulbs[frames - 2].m_position.x;
			double deltaY = m_vecBulbs[frames - 1].m_position.y - m_vecBulbs[frames - 2].m_position.y;

			predictedPosition.x = m_vecBulbs.back().m_position.x + deltaX;
			predictedPosition.y = m_vecBulbs.back().m_position.y + deltaY;
		}
		else if (frames == 3)
		{
			double sumOfXChanges = 
				((m_vecBulbs[frames - 1].m_position.x - m_vecBulbs[frames - 2].m_position.x) * 2) +
				((m_vecBulbs[frames - 2].m_position.x - m_vecBulbs[frames - 3].m_position.x) * 1);
			
			double sumOfYChanges = 
				((m_vecBulbs[frames - 1].m_position.y - m_vecBulbs[frames - 2].m_position.y) * 2) +
				((m_vecBulbs[frames - 2].m_position.y - m_vecBulbs[frames - 3].m_position.y) * 1);

			double deltaX = sumOfXChanges / 3.0;
			double deltaY = sumOfYChanges / 3.0;
			
			predictedPosition.x = m_vecBulbs.back().m_position.x + deltaX;
			predictedPosition.y = m_vecBulbs.back().m_position.y + deltaY;
		}
		else
		{
			double sumOfXChanges = 
				((m_vecBulbs[frames - 1].m_position.x - m_vecBulbs[frames - 2].m_position.x) * 4) +
				((m_vecBulbs[frames - 2].m_position.x - m_vecBulbs[frames - 3].m_position.x) * 2) +
				((m_vecBulbs[frames - 3].m_position.x - m_vecBulbs[frames - 4].m_position.x) * 1);
			
			double sumOfYChanges = 
				((m_vecBulbs[frames - 1].m_position.y - m_vecBulbs[frames - 2].m_position.y) * 4) +
				((m_vecBulbs[frames - 2].m_position.y - m_vecBulbs[frames - 3].m_position.y) * 2) +
				((m_vecBulbs[frames - 3].m_position.y - m_vecBulbs[frames - 4].m_position.y) * 1);

			double deltaX = sumOfXChanges / 6.0;
			double deltaY = sumOfYChanges / 6.0;

			predictedPosition.x = m_vecBulbs.back().m_position.x + deltaX;
			predictedPosition.y = m_vecBulbs.back().m_position.y + deltaY;
		}
		return cv::Point2d(predictedPosition);
	}


	void LightTracker::imshow(
		bool previousFrame,
		bool currentFrame, 
		bool lightThresh1, 
		bool lightThresh2, 
		bool lightMask, 
		bool bulb, 
		bool contoursMoments)
	{
		if (previousFrame)	cv::imshow(m_sPreviousFrame, m_imgPreviousFrame);
		else				cv::destroyWindow(m_sPreviousFrame);
		if (currentFrame)	cv::imshow(m_sCurrentFrame, m_imgCurrentFrame);
		else				cv::destroyWindow(m_sCurrentFrame);
		if (lightThresh1)	cv::imshow(m_sLightThresh1, m_imgLightThresh1);
		else				cv::destroyWindow(m_sLightThresh1);
		if (lightThresh2)	cv::imshow(m_sLightThresh2, m_imgLightThresh2);
		else				cv::destroyWindow(m_sLightThresh2);
		if (lightMask) 		cv::imshow(m_sLightMask, m_imgLightMask);
		else				cv::destroyWindow(m_sLightMask);
		if (bulb)			cv::imshow(m_sBulb, m_imgBulb_detectBulb);
		else				cv::destroyWindow(m_sBulb);
		if (contoursMoments)
		{
			m_imgContoursMoments = cv::Mat(m_imgCanny_detectBulb.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			cv::Scalar color = cv::Scalar(167, 151, 0);

			for (int i = 0; i < m_vecContours_detectBulb.size(); i++)
			{
				cv::drawContours(m_imgContoursMoments, m_vecContours_detectBulb, i, color, 2, 8, m_vecHierarchy_detectBulb, 0, cv::Point());
				cv::circle(m_imgContoursMoments, m_vecCentralMoments_detectBulb[i], 4, -1, 0);
			}
			cv::imshow(m_sContoursMoments, m_imgContoursMoments);
		}
		else				cv::destroyWindow(m_sContoursMoments);
	}
	
	

	Bulb::Bulb(cv::Point2d& pos, int32_t frameNumber, bool visible)
		: m_position(pos), m_frameNumber(frameNumber), m_visible(visible), m_velocity(0.0), m_direction(cv::Vec2d(0,0))
	{
	}
	//set motion relative to bulb in previous frame
	void Bulb::setMotion(const bs::Bulb* b)
	{
		m_direction = m_position - b->m_position;
		m_velocity = cv::sqrt(cv::pow(m_direction[0], 2) + cv::pow(m_direction[1], 2));
	}

	std::ostream & operator<<(std::ostream &out, const Bulb &b)
	{
		out << "dir: " << b.m_direction << "vel: " << b.m_velocity << std::endl;
		return out;
	}

}