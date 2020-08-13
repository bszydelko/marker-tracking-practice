#include "opencv2/opencv.hpp"
#include "CaptureYUV.h"
#include "MarkerTracker.h"
#include <string>



int main(int argc, char** argv) //TODO - provide support for handling main parameters
{
	std::string sCamNum = "31";
	std::string sSequence = "cam" + sCamNum + "_1920x1080.yuvdist.yuv";
	std::string sMask = "mask_cam" + sCamNum + "_1920x1080.yuvdist.yuv";
	std::string sPath = "E:\\_SEQ\\" + sCamNum +"\\";
	//std::string sPath = "";

	int32_t width = 1920;
	int32_t height = 1080;
	int32_t chromaSubsampling = 420;

	if (argc > 1)
	{
		sPath = "";
		sSequence = argv[1];
		sMask = argv[2];
		width = std::stoi(argv[3]);
		height = std::stoi(argv[4]);
		chromaSubsampling = std::stoi(argv[5]);

	}
	bs::CaptureYUV mask(sPath + sMask, width, height, chromaSubsampling, 1);
	if (!mask.isOpened()) return 0;

	bs::CaptureYUV forwardVideo(sPath + sSequence, width, height, chromaSubsampling, 1);
	if (!forwardVideo.isOpened()) return 0;

	std::cout << "forward: " << std::endl;
	bs::MarkerTracker forwardTracker(&forwardVideo, &mask);
	forwardTracker.start();
	std::vector<cv::Point2d> forward_points = forwardTracker.getPoints();

	bs::CaptureYUV reverseVideo(sPath + sSequence, width, height, chromaSubsampling, -1);
	if (!forwardVideo.isOpened()) return 0;

	std::cout << "reverse:" << std::endl;
	bs::MarkerTracker reverseTracker(&reverseVideo, &mask);
	reverseTracker.start();
	std::vector<cv::Point2d> reverse_points = reverseTracker.getPoints();
	std::reverse(reverse_points.begin(), reverse_points.end());



	//analyze points
	auto distance = [](const cv::Point2d& pt1, const cv::Point2d& pt2)
	{
		cv::Point2d diff = pt1 - pt2;
		return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
	};

	std::vector<cv::Point2d> final_points;
	final_points.reserve(forward_points.size());
	double_t avg_distance = 0.0;
	int count = 0;

	for (size_t i = 0; i < forward_points.size(); i++)
	{
		if (forward_points[i] == cv::Point2d(-1, -1) || reverse_points[i] == cv::Point2d(-1, -1))
			continue;
		avg_distance += distance(forward_points[i], reverse_points[i]);
		count++;
	}
	avg_distance /= count;

	for (size_t i = 0; i < forward_points.size(); i++)
	{
		if (forward_points[i] == cv::Point2d(-1, -1) || reverse_points[i] == cv::Point2d(-1, -1) || distance(forward_points[i], reverse_points[i]) >= avg_distance * 2)
			final_points.emplace_back(cv::Point2d(-1,-1));
		else 
			final_points.push_back(forward_points[i]);
		
	}

	std::ofstream file_final_pos(sPath + "final_pos.txt", std::ios::out);

	for (size_t i = 0; i < forward_points.size(); i++)
	{
		file_final_pos << i << "\t" << final_points[i].x << "\t" << final_points[i].y << std::endl;
	}

	file_final_pos.close();



	return 0;
}
