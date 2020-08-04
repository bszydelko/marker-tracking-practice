#include "MarkerTracker.h"

namespace bs
{
	MarkerTracker::MarkerTracker(bs::VideoCaptureYUV* video)
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
		cv::resizeWindow(m_sBulb, cv::Size(250,250));

		cv::namedWindow(m_sContoursMoments, cv::WINDOW_KEEPRATIO);
		cv::resizeWindow(m_sContoursMoments, video->getResolution());

		m_kernelDilate_thresholdLights = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
		kernelDilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));

	}

	int32_t MarkerTracker::start()
	{
		//thresh1
		video->read(imgCurrentFrame);
		threshold_lights(imgCurrentFrame, imgLightThresh1);

		//thresh2
		video->read(imgCurrentFrame);
		threshold_lights(imgCurrentFrame, imgLightThresh2);

		create_light_mask(imgLightThresh1, imgLightThresh2, imgLightMask);

		//imshow triggers
		bool previousFrame = 0;
		bool currentFrame = 1;
		bool lightThresh1 = 0;
		bool lightThresh2 = 0;
		bool lightMask = 1;
		bool bulb = 1;
		bool contoursMoments = 0;

		cv::Mat currentMaskRegion;
		cv::Mat currentFrameRegion;
		cv::Point2d marker;
		cv::Point2d predictedMarker;
		cv::Rect2d predictedRegion = cv::Rect2d(cv::Point2d(0,0), imgCurrentFrame.size());
		cv::Rect frameBound = cv::Rect(cv::Point(0, 0), imgCurrentFrame.size());
		std::vector<cv::Point2d> vecCurrentCentralMoments;
		std::vector < std::vector<cv::Point>> vecCurrentContours;
		bs::MARKER_STATE whole_frame_state;
		bs::MARKER_STATE region_frame_state;
		Marker prevMarker;

		cv::waitKey(WAIT_TIME);


		while (video->read(imgCurrentFrame))
		{
			if (cv::waitKey(5) == 27) break; 

			imgCurrentFrame.copyTo(imgRawFrame);
			//new stuff

			
			
			whole_frame_state = process_frame(
				imgCurrentFrame, 
				imgLightMask, 
				vecCurrentContours, 
				vecCurrentCentralMoments);
			

			switch (whole_frame_state)
			{
			case BULB_VISIBLE: //whole image
				//process frame in range
				predictedMarker = predict_average();
				predictedRegion = create_region(predictedMarker);

				currentMaskRegion = imgLightMask(predictedRegion);
				currentFrameRegion = imgRawFrame(predictedRegion);

				region_frame_state = process_frame(
					currentFrameRegion,
					currentMaskRegion,
					vecCurrentContours,
					vecCurrentCentralMoments);

				switch (region_frame_state) //region
				{
				case BULB_VISIBLE:
					marker = select_marker(
						vecCurrentCentralMoments,
						vecCurrentContours,
						predictedMarker,
						predictedRegion,
						currentMaskRegion);

					break;
				case BULB_NOT_VISIBLE: //find in larger region by expanding current region
					int32_t expand = 3;
					int32_t max_attempt = 4;
					int32_t attempt = 0;
					MARKER_STATE state = BULB_NOT_VISIBLE;

					while (attempt < max_attempt && state == BULB_NOT_VISIBLE)
					{

						predictedRegion = create_region(predictedMarker, expand);
						currentMaskRegion = imgLightMask(predictedRegion);
						currentFrameRegion = imgRawFrame(predictedRegion);
						state = process_frame(
							currentFrameRegion,
							currentMaskRegion,
							vecCurrentContours,
							vecCurrentCentralMoments);
						expand += 1;
						attempt++;
					}
					if (state == BULB_VISIBLE) {
						marker = select_marker(
							vecCurrentCentralMoments,
							vecCurrentContours,
							predictedMarker,
							predictedRegion,
							currentMaskRegion);
					}
					else
					{
						std::cout << "region expanded not successful" << std::endl;
					}

					break;
				}

				

				if (m_vecMarker.size() > 0) prevMarker = m_vecMarker.back();
				m_vecMarker.emplace_back(bs::Marker(marker, video->getFrameID()));
				if (m_vecMarker.size() > 1)m_vecMarker.back().setMotion(&prevMarker);
				
				std::cout << video->getFrameID() << " " << marker << std::endl;
				break;

			case BULB_NOT_VISIBLE:

				if(!frameBound.contains(predictedMarker))
					std::cout << video->getFrameID() << "out of frame" << std::endl;
				break;
			}


			//new stuff
			
			cv::circle(imgCurrentFrame, marker, 20, cv::Scalar(0, 255, 0), 3);
			cv::rectangle(imgCurrentFrame, predictedRegion, cv::Scalar(255, 0, 255), 3);
			cv::drawMarker(imgCurrentFrame, predictedMarker, cv::Scalar(0, 0, 255), cv::MarkerTypes::MARKER_CROSS, 30, 3);
			imshow(previousFrame, currentFrame, lightThresh1, lightThresh2, lightMask, bulb, contoursMoments);

			imgCurrentFrame.copyTo(imgPreviousFrame);
			cv::waitKey(WAIT_TIME);

		}

		return 1;
	}

	MarkerTracker::~MarkerTracker()
	{

	}

	void MarkerTracker::threshold_lights(const cv::Mat& frame, cv::Mat& imgThresh)
	{
		//cv::Mat m_kernelErode_detectLight = cv::getStructuringElement()

		cv::cvtColor(frame, imgThresh, cv::COLOR_BGR2GRAY);
		cv::blur(imgThresh, imgThresh, cv::Size(9, 9));
		cv::dilate(imgThresh, imgThresh, m_kernelDilate_thresholdLights);
		cv::threshold(imgThresh, imgThresh, m_thresh_thresholdLights, 255, cv::THRESH_TOZERO);
	}

	void MarkerTracker::create_light_mask(
		const cv::Mat& frame1, 
		const cv::Mat& frame2, 
		cv::Mat& mask)
	{
		cv::Rect bound(10, 10, frame1.size().width - 20, frame1.size().height - 20);
		cv::Mat maskBound = 255 * cv::Mat::ones(frame1.size(), CV_8UC1);
		cv::rectangle(maskBound, bound, cv::Scalar(0), -1);
		cv::bitwise_and(frame1, frame2, mask);

		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::erode(maskBound, maskBound, kernel);
		cv::bitwise_or(maskBound, mask, mask);
	}


	MARKER_STATE MarkerTracker::process_frame(
		const cv::Mat& frame, 
		const cv::Mat& mask, 
		std::vector<std::vector<cv::Point>> &vecCurrentContour,
		std::vector<cv::Point2d> &vecCurrentCentralMoments,
		int32_t thresh,
		bool bBlur)
	{
		threshold_lights(frame, imgThresh);
		cv::absdiff		(imgThresh, mask, imgBulb);
		if(bBlur) cv::GaussianBlur(imgBulb, imgBulb, cv::Size(3, 3), 0);
		cv::threshold	(imgBulb, imgBulb, thresh, 255, cv::THRESH_BINARY);
		cv::dilate		(imgBulb, imgBulb, kernelDilate);

		//check if another object cover mask
		cv::Mat kernel_e = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
		cv::Mat kernel_d = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

		cv::Mat bulb;
		imgBulb.copyTo(bulb);
		cv::erode(bulb, bulb, kernel_e);
		for (size_t r = 0; r < frame.rows; r++)
		{
			for (size_t c = 0; c < frame.cols; c++)
			{
				if (mask.at<uint8_t>(r, c) && imgBulb.at<uint8_t>(r, c))
					bulb.at<uint8_t>(r, c) = 0;
				
			}
		}

		cv::dilate(bulb, bulb, kernel_d);
		//  cv::GaussianBlur(bulb, bulb, cv::Size(3, 3), 0);
		bulb.copyTo(imgBulb);
		//check if another object cover mask

		//contours

		vecCurrentContour.clear();
		vecCurrentCentralMoments.clear();
		cv::Canny(
			imgBulb, 
			m_imgCanny, 50, 150, 3);
		cv::findContours(
			m_imgCanny, 
			vecCurrentContour, 
			m_vecContourHierarchy, 
			cv::RETR_EXTERNAL, 
			cv::CHAIN_APPROX_NONE, 
			cv::Point(0, 0));

		m_vecMoments.clear();
		m_vecMoments.reserve(vecCurrentContour.size());
		vecCurrentCentralMoments.clear();
		vecCurrentCentralMoments.reserve(vecCurrentContour.size());
		//contours

		for (int i = 0; i < vecCurrentContour.size(); i++)
		{
			m_vecMoments.emplace_back(
				cv::moments(vecCurrentContour[i], false)); 		//moments for each blob / contour
			
			vecCurrentCentralMoments.emplace_back( 
				cv::Point2d(											//center of each blob / contour
					m_vecMoments[i].m10 / cv::contourArea(vecCurrentContour[i]), 
					m_vecMoments[i].m01 / cv::contourArea(vecCurrentContour[i])));
		}

		if (vecCurrentContour.size() == 0) return BULB_NOT_VISIBLE;
		else return BULB_VISIBLE;
	}

	cv::Rect2d MarkerTracker::create_region(const cv::Point2d& predictedMarker, const int32_t expand)
	{
		if (m_vecMarker.size() == 0)
		{
			return cv::Rect2d(0, 0, imgCurrentFrame.size().width, imgCurrentFrame.size().height);
		}

		auto avgDim = [&](int n)
		{
			double sum = 0.0;
			if (n > m_vecMarker.size()) n = m_vecMarker.size();
			if(n == 1) sum = std::sqrt(
				std::pow(m_vecMarker.back().m_position.x, 2) +
				std::pow(m_vecMarker.back().m_position.y, 2));
			else sum = std::sqrt(
				std::pow(m_vecMarker.back().m_position.x - predictedMarker.x, 2) +
				std::pow(m_vecMarker.back().m_position.y - predictedMarker.y, 2));
			auto it = m_vecMarker.rbegin();

			for (int i = 0; i < n - 1; i++)
			{

				sum += (std::sqrt(
					std::pow((*it).m_position.x - (*(it+1)).m_position.x, 2) +
					std::pow((*it).m_position.y - (*(it+1)).m_position.y, 2)));
				sum *= 2.0;
				//sum += (*it).m_velocity;
				//sum += std::abs((*it).m_direction[0]) + std::abs((*it).m_direction[1]);
				++it;
			}
			return sum / (double)n;
 

		};

		double dim = avgDim(4) * expand;
		cv::Rect2d region(predictedMarker.x - dim / 2.0, predictedMarker.y - dim / 2.0, dim, dim);

		if (region.width > imgCurrentFrame.size().width)
			region.width = imgCurrentFrame.size().width;
		if (region.height > imgCurrentFrame.size().height)
			region.height = imgCurrentFrame.size().height;

		if (region.x < 0) 
			region.x = 0;
		if (region.x + region.width > imgCurrentFrame.size().width) 
			region.x = imgCurrentFrame.size().width - region.width;
		if (region.y < 0)
			region.y = 0;
		if (region.y + region.height > imgCurrentFrame.size().height)
			region.y = imgCurrentFrame.size().height - region.height;

	

		return cv::Rect2d(region);
	}

	cv::Point2d MarkerTracker::select_marker(
		const std::vector<cv::Point2d>& vecCurrentCentralMoments,
		const std::vector<std::vector<cv::Point>>& vecCurrentContours, 
		const cv::Point2d& predictedMarker, 
		const cv::Rect2d& predictedRegion,
		const cv::Mat& currentMaskRegion)
	{
		//precondition: vecMarker.size() >= 1
		auto distance = [](const cv::Point2d& pt1, const cv::Point2d& pt2)
		{
			cv::Point diff = pt1 - pt2;
			return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
		};
		
		std::vector<cv::Point2d> vecMarkerInRegion;
		cv::Point2d retPoint(-1,-1);

		if (vecCurrentCentralMoments.size() > 1)
		{
			for (int i = 0; i < vecCurrentCentralMoments.size(); i++) //checks collision for each marker in region
			{
				if(region_contains_contour(predictedRegion, vecCurrentContours[i])) //NOPE NOPE NOPE, it actually checks contour in region XDDDD fix it
					if (!bulbVsMask(vecCurrentContours[i], currentMaskRegion))
						vecMarkerInRegion.push_back(vecCurrentCentralMoments[i]);
			}

			/*double dist2 = 0.0;
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

			}*/

			double area2 = 0.0;
			double area1 = cv::contourArea(vecCurrentContours[0]);
			int bigger = 0;

			for (int i = 1; i < vecMarkerInRegion.size(); i++)
			{
				area2 = cv::contourArea(vecCurrentContours[i]);
				if (area2 > area1)
				{
					area1 = area2;
					bigger = i;
				}
			}


			//temporary fix
			if (vecMarkerInRegion.size() == 0) {
				return retPoint;
			}


			retPoint = cv::Point2d(
				predictedRegion.tl().x + vecMarkerInRegion[bigger].x,
				predictedRegion.tl().y + vecMarkerInRegion[bigger].y);

		}
		else
		{
			cv::Point2d fixedCentralMoment = vecCurrentCentralMoments[0];
			fixedCentralMoment.x += predictedRegion.tl().x;
			fixedCentralMoment.y += predictedRegion.tl().y;

			if(region_contains_contour(predictedRegion, vecCurrentContours[0]))
			{
				if (!bulbVsMask(vecCurrentContours[0], currentMaskRegion))
					retPoint = fixedCentralMoment;
			}
			else
			{
				double dist = distance(vecCurrentCentralMoments[0], predictedMarker);
			}


		}


		return retPoint;
	}

	bool MarkerTracker::region_contains_contour(const cv::Rect2d& region, const std::vector<cv::Point>& contour)
	{
		bool retval = true;
		for ( auto point : contour) {
			point.x += region.x; //fix region offset 
			point.y += region.y;
			if (!region.contains(point))
			{
				retval = false;
				break;
			}
		}
		return retval;
	}


	bool MarkerTracker::bulbVsMask(const std::vector<cv::Point>& contour, const cv::Mat& mask)
	{
		for (const auto& c : contour)
		{
			if (mask.at<uint8_t>(c) >= 1) 
				return true;
		}
		return false;
	}

	double MarkerTracker::distance(const cv::Point2d& p1, const cv::Point2d& p2) const
	{
		return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y,2));
	}

	cv::Point2d MarkerTracker::predict_average()
	{
		cv::Point2d predictedPosition;
		int frames = m_vecMarker.size();
		
		if (frames == 0) return cv::Point2d(imgCurrentFrame.size() / 2);
		else if (frames == 1) return m_vecMarker[0].m_position;
		else if (frames == 2)
		{
			double deltaX = m_vecMarker[frames - 1].m_position.x - m_vecMarker[frames - 2].m_position.x;
			double deltaY = m_vecMarker[frames - 1].m_position.y - m_vecMarker[frames - 2].m_position.y;

			predictedPosition.x = m_vecMarker.back().m_position.x + deltaX;
			predictedPosition.y = m_vecMarker.back().m_position.y + deltaY;
		}
		else if (frames == 3)
		{
			double sumOfXChanges = 
				((m_vecMarker[frames - 1].m_position.x - m_vecMarker[frames - 2].m_position.x) * 2) +
				((m_vecMarker[frames - 2].m_position.x - m_vecMarker[frames - 3].m_position.x) * 1);
			
			double sumOfYChanges = 
				((m_vecMarker[frames - 1].m_position.y - m_vecMarker[frames - 2].m_position.y) * 2) +
				((m_vecMarker[frames - 2].m_position.y - m_vecMarker[frames - 3].m_position.y) * 1);

			double deltaX = sumOfXChanges / 3.0;
			double deltaY = sumOfYChanges / 3.0;
			
			predictedPosition.x = m_vecMarker.back().m_position.x + deltaX;
			predictedPosition.y = m_vecMarker.back().m_position.y + deltaY;
		}
		else
		{
			double sumOfXChanges = 
				((m_vecMarker[frames - 1].m_position.x - m_vecMarker[frames - 2].m_position.x) * 4) +
				((m_vecMarker[frames - 2].m_position.x - m_vecMarker[frames - 3].m_position.x) * 2) +
				((m_vecMarker[frames - 3].m_position.x - m_vecMarker[frames - 4].m_position.x) * 1);
			
			double sumOfYChanges = 
				((m_vecMarker[frames - 1].m_position.y - m_vecMarker[frames - 2].m_position.y) * 4) +
				((m_vecMarker[frames - 2].m_position.y - m_vecMarker[frames - 3].m_position.y) * 2) +
				((m_vecMarker[frames - 3].m_position.y - m_vecMarker[frames - 4].m_position.y) * 1);

			double deltaX = sumOfXChanges / 6.0;
			double deltaY = sumOfYChanges / 6.0;

			predictedPosition.x = m_vecMarker.back().m_position.x + deltaX;
			predictedPosition.y = m_vecMarker.back().m_position.y + deltaY;
		}

		return predictedPosition;


	}


	void MarkerTracker::imshow(
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
			/*imgContoursMoments = cv::Mat(m_imgCanny.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			cv::Scalar color = cv::Scalar(167, 151, 0);

			for (int i = 0; i < vecContour.size(); i++)
			{
				cv::drawContours(imgContoursMoments, vecContour, i, color, 2, 8, m_vecContourHierarchy, 0, cv::Point());
				cv::circle(imgContoursMoments, vecCentralMoments[i], 4, -1, 0);
			}
			cv::imshow(m_sContoursMoments, imgContoursMoments);*/
		}
		else				cv::destroyWindow(m_sContoursMoments);
	}
	
	

	Marker::Marker(cv::Point2d& pos, int32_t frameNumber, bool visible)
		: m_position(pos), m_frameNumber(frameNumber), m_visible(visible), m_velocity(0.0), m_direction(cv::Vec2d(0,0))
	{
	}
	//set motion relative to bulb in previous frame
	void Marker::setMotion(const bs::Marker* prevMarker)
	{
		m_direction = m_position - prevMarker->m_position;
		m_velocity = cv::sqrt(cv::pow(m_direction[0], 2) + cv::pow(m_direction[1], 2));
	}

	std::ostream & operator<<(std::ostream &out, const Marker &b)
	{
		out << "dir: " << b.m_direction << "vel: " << b.m_velocity << std::endl;
		return out;
	}

}