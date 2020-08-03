#include "LightTracker.h"

namespace bs
{
	LightTracker::LightTracker(bs::VideoCaptureYUV* video)
		: video(video)
	{
		//initialize windows
		cv::namedWindow(m_sPreviousFrame, cv::WINDOW_KEEPRATIO);
		cv::resizeWindow(m_sPreviousFrame, video->getResolution() / 2);

		cv::namedWindow(m_sCurrentFrame, cv::WINDOW_KEEPRATIO);
		cv::resizeWindow(m_sCurrentFrame, video->getResolution() / 2);

		cv::namedWindow(m_sLightThresh1, cv::WINDOW_KEEPRATIO);
		cv::resizeWindow(m_sLightThresh1, video->getResolution() / 3);
		cv::namedWindow(m_sLightThresh2, cv::WINDOW_KEEPRATIO);
		cv::resizeWindow(m_sLightThresh2, video->getResolution() / 3);

		cv::namedWindow(m_sLightMask, cv::WINDOW_KEEPRATIO);
		cv::resizeWindow(m_sLightMask, video->getResolution() / 3);

		cv::namedWindow(m_sBulb, cv::WINDOW_KEEPRATIO);
		cv::resizeWindow(m_sBulb, video->getResolution() / 2);

		cv::namedWindow(m_sContoursMoments, cv::WINDOW_KEEPRATIO);
		cv::resizeWindow(m_sContoursMoments, video->getResolution());

		m_kernelDilate_thresholdLights = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
		kernelDilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));

	}

	int32_t LightTracker::start()
	{


		//thresh1
		video->read(imgCurrentFrame);
		thresholdLights(imgCurrentFrame, imgLightThresh1);

		//thresh2
		video->read(imgCurrentFrame);
		thresholdLights(imgCurrentFrame, imgLightThresh2);

		createLightMask(imgLightThresh1, imgLightThresh2, imgLightMask);

	
		//imshow triggers
		bool previousFrame = 0;
		bool currentFrame = 1;
		bool lightThresh1 = 0;
		bool lightThresh2 = 0;
		bool lightMask = 0;
		bool bulb = 1;
		bool contoursMoments = 1;

		cv::Point2d marker;
		cv::Point2d predictedMarker;
		cv::Rect predictedRegion;
		std::vector<cv::Point2d> vecCurrentMarker;
		std::vector < std::vector<cv::Point>> vecCurrentContour;
		MARKER_STATE state;

		while (video->read(imgCurrentFrame))
		{
			if (cv::waitKey(5) == 27) break;

			imgCurrentFrame.copyTo(imgRawFrame);


			//new stuff

			state = find_marker();
			predictedMarker = predictAverage();

			switch (state)
			{
			case BULB_VISIBLE:
				
				marker = select_marker(vecCurrentMarker, predictedMarker, predictedRegion);

				break;

			case BULB_NOT_VISIBLE:


				break;
			}


			//new stuff
			
			imshow(previousFrame, currentFrame, lightThresh1, lightThresh2, lightMask, bulb, contoursMoments);
			std::cout << video->getFrameID() << m_bulbPos << std::endl;

			imgCurrentFrame.copyTo(imgPreviousFrame);
			cv::waitKey(WAIT_TIME);

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


	MARKER_STATE LightTracker::find_marker()
	{
		thresholdLights	(imgRawFrame, imgThresh);
		cv::absdiff		(imgThresh, imgLightMask, imgBulb);
		cv::GaussianBlur(imgBulb, imgBulb, cv::Size(3, 3), 0);
		cv::threshold	(imgBulb, imgBulb, 220, 255, cv::THRESH_BINARY);
		cv::dilate		(imgBulb, imgBulb, kernelDilate);

		//check if another object cover mask
		cv::Mat kernel_e = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
		cv::Mat kernel_d = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

		cv::Mat bulb;
		imgBulb.copyTo(bulb);
		cv::erode(bulb, bulb, kernel_e);
		bool retVal = false;
		for (size_t r = 0; r < imgRawFrame.rows; r++)
		{
			for (size_t c = 0; c < imgRawFrame.cols; c++)
			{
				if (imgLightMask.at<uint8_t>(r, c) && imgBulb.at<uint8_t>(r, c))
				{
					retVal = true;
					bulb.at<uint8_t>(r, c) = 0;
				}
			}
		}

		cv::dilate(bulb, bulb, kernel_d);
		bulb.copyTo(imgBulb);
		//check if another object cover mask

		//contours
		cv::Canny(
			imgBulb, 
			imgCanny, 50, 150, 3);
		cv::findContours(
			imgCanny, 
			vecContour, 
			vecHierarchy, 
			cv::RETR_EXTERNAL, 
			cv::CHAIN_APPROX_NONE, 
			cv::Point(0, 0));

		vecMoments.clear();
		vecMoments.reserve(vecContour.size());
		vecMarker.clear();
		vecMarker.reserve(vecContour.size());
		//contours

		for (int i = 0; i < vecContour.size(); i++)
		{
			vecMoments[i] = 
				cv::moments(vecContour[i], false); 		//moments for each blob / contour
			vecMarker[i] = 
				cv::Point2d(											//center of each blob / contour
					vecMoments[i].m10 / vecMoments[i].m00, 
					vecMoments[i].m01 / vecMoments[i].m00); 		
		}

		if (vecContour.size() == 0) return BULB_NOT_VISIBLE;
		else return BULB_VISIBLE;
	}

	cv::Point2d LightTracker::select_marker(const cv::Point2d& predictedMarker, const cv::Rect& predictedRegion)
	{
		//precondition: vecMarker.size() >= 1
		auto distance = [](const cv::Point2d& pt1, const cv::Point2d& pt2)
		{
			cv::Point diff = pt1 - pt2;
			return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
		};
		
		std::vector<cv::Point2d> vecMarkerInRegion;
		cv::Point2d retPoint(-1,-1);

		if (vecMarker.size() > 1)
		{
			for (int i = 0; i < vecMarker.size(); i++) //checks collision for each marker in region
			{
				if (predictedRegion.contains(vecMarker[i]))
					if (!bulbVsMask(vecContour[i], imgLightMask))
						vecMarkerInRegion.push_back(vecMarker[i]);
			}

			double dist2 = 0.0;
			double dist1 = distance(vecMarkerInRegion[0], predictedMarker);
			int closer = 0;

			for (int i = 1; i < vecMarkerInRegion.size(); i++)
			{
				dist2 = distance(vecMarkerInRegion[i], predictedMarker);
				if (dist2 < dist1)
				{
					dist2 = dist1;
					closer = i;
				}

			}

			retPoint = vecMarkerInRegion[closer];

		}
		else
		{
			if (predictedRegion.contains(vecMarker[0]))
			{
				if (!bulbVsMask(vecContour[0], imgLightMask))
					retPoint = vecMarker[0];
			}
			else
			{
				double dist = distance(vecMarker[0], predictedMarker);
			}


		}


		return retPoint;
	}

	cv::Point2d LightTracker::find_marker_in_close_range(const cv::Mat& frame, const cv::Mat& mask, const cv::Point2d marker)
	{
		cv::waitKey(WAIT_TIME);
		auto avgDim = [&](int n)
		{
			if (n > m_vecBulbs.size()) n = m_vecBulbs.size();
			double sum = std::sqrt(
				std::pow(m_vecBulbs.back().m_position.x - marker.x, 2) +
				std::pow(m_vecBulbs.back().m_position.y - marker.y, 2));
			auto it = m_vecBulbs.end();
			
			for (int i = 0; i < n - 1; i++)
			{
				
				sum += (std::sqrt(
					std::pow((*it).m_position.x - (*(it - 1)).m_position.x, 2) +
					std::pow((*it).m_position.y - (*(it - 1)).m_position.y, 2)));
				it--;
			}
			return sum / (double)n / 2.0;
		};
		
		double dim = avgDim(4);
		cv::Rect bound(marker.x - dim / 2, marker.y - dim / 2, dim, dim);
		cv::Mat frame_roi = frame(bound);
		cv::Mat mask_roi = mask(bound);
		cv::Mat frame_roi_thresh;
		cv::Mat frame_roi_bulb;
		cv::Mat frame_roi_canny;
		thresholdLights(frame_roi, frame_roi_thresh);
		cv::absdiff(frame_roi_thresh, mask_roi, frame_roi_bulb);
		//cv::GaussianBlur(frame_roi_bulb, frame_roi_bulb, cv::Size(3, 3), 0);
		cv::threshold(frame_roi_bulb, frame_roi_bulb, 200, 255, cv::THRESH_BINARY);
		cv::dilate(frame_roi_bulb, frame_roi_bulb, kernelDilate);

		std::vector<std::vector<cv::Point>> vecContours;
		std::vector<cv::Vec4i>				vecHierarchy;
		std::vector<cv::Moments>			vecMoments;
		std::vector<cv::Point2f>			vecCentralMoments;

		cv::Canny(
			frame_roi_bulb,
			frame_roi_canny, 50, 150, 3);
		cv::findContours(
			frame_roi_canny,
			vecContours,
			vecHierarchy,
			cv::RETR_TREE,
			cv::CHAIN_APPROX_SIMPLE,
			cv::Point(0, 0));

		vecMoments.reserve(vecContours.size());
		vecCentralMoments.reserve(vecContours.size());

		for (int i = 0; i < vecContours.size(); i++) {
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

		if (vecContours.size() > 1) {
			area1 = cv::contourArea(vecContours[0]);
			retPoint = vecCentralMoments[0];

			for (int i = 1; i < vecContours.size(); i++) {
				area2 = cv::contourArea(vecContours[i]);
				if (area2 > area1) {
					retPoint = vecCentralMoments[i];
					area1 = area2;
				}
			}
		}

		cv::imshow("roi", frame_roi_bulb);
		cv::waitKey(WAIT_TIME);
		return retPoint;

		
	}

	cv::Point2d LightTracker::find_marker_in_first_frame(const cv::Mat& frame, const cv::Mat& mask)
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
		cv::dilate		(frame_bulb, frame_bulb, kernelDilate);

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

	double LightTracker::distance(const cv::Point2d& p1, const cv::Point2d& p2) const
	{
		return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y,2));
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

		return predictedPosition;


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
		if (previousFrame)	cv::imshow(m_sPreviousFrame, imgPreviousFrame);
		else				cv::destroyWindow(m_sPreviousFrame);
		if (currentFrame)	cv::imshow(m_sCurrentFrame, imgCurrentFrame);
		else				cv::destroyWindow(m_sCurrentFrame);
		if (lightThresh1)	cv::imshow(m_sLightThresh1, imgLightThresh1);
		else				cv::destroyWindow(m_sLightThresh1);
		if (lightThresh2)	cv::imshow(m_sLightThresh2, imgLightThresh2);
		else				cv::destroyWindow(m_sLightThresh2);
		if (lightMask) 		cv::imshow(m_sLightMask, imgLightMask);
		else				cv::destroyWindow(m_sLightMask);
		if (bulb)			cv::imshow(m_sBulb, imgBulb);
		else				cv::destroyWindow(m_sBulb);
		if (contoursMoments)
		{
			imgContoursMoments = cv::Mat(imgCanny.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			cv::Scalar color = cv::Scalar(167, 151, 0);

			for (int i = 0; i < vecContour.size(); i++)
			{
				cv::drawContours(imgContoursMoments, vecContour, i, color, 2, 8, vecHierarchy, 0, cv::Point());
				cv::circle(imgContoursMoments, vecMarker[i], 4, -1, 0);
			}
			cv::imshow(m_sContoursMoments, imgContoursMoments);
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