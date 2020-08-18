#include "opencv2/opencv.hpp"
#include "CaptureYUV.h"
#include "MarkerTracker.h"
#include <string>




int main(int argc, char** argv) 
{
	cv::String keys =
		"{@seq_num   |	         | sequence list path}"
		"{@frame_num	   |	         | sequence list path}"
		"{@path_seq_list	   |	         | sequence list path}"
		"{@path_mask_list	   |	         | sequence list path}"
		"{@width	   |	         | sequence list path}"
		"{@height	   |	         | sequence list path}"
		"{@chromaSubsampling	   |	         | sequence list path}";
	
	cv::CommandLineParser parser(argc, argv, keys);

	uint32_t seq_num = parser.get<uint32_t>(0);
	uint32_t frame_num = parser.get<uint32_t>(1);
	std::string path_seq_list = parser.get<std::string>(2);
	std::string path_mask_list = parser.get<std::string>(3);
	uint32_t width = parser.get<uint32_t>(4);
	uint32_t height = parser.get<uint32_t>(5);
	uint32_t chromaSubsampling = parser.get<uint32_t>(6);

	std::ifstream file_seq_list(path_seq_list, std::ios::in);
	std::ifstream file_mask_list(path_mask_list, std::ios::in);


	if (!file_seq_list.is_open()) {
		std::cout << "Sequence list does not exist!" << std::endl;
		return EXIT_FAILURE;
	}
	if (!file_mask_list.is_open()) {
		std::cout << "Mask list does not exist!" << std::endl;
		return EXIT_FAILURE;
	}
	
	auto distance = [](const cv::Point2d& pt1, const cv::Point2d& pt2)
	{
		cv::Point2d diff = pt1 - pt2;
		return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
	};

	std::string sSequence;
	std::string sMask;

	
	std::vector<cv::Point2d> final_points;
	std::vector<cv::Point2d> forward_points;
	std::vector<cv::Point2d> reverse_points;

	const cv::Point2d NOT_DETECTED(-1, -1);
	cv::Point2d** points = new cv::Point2d*[frame_num];
	for (size_t i = 0; i < frame_num; i++)
		points[i] = new cv::Point2d[seq_num];

	uint32_t current_seq = 0;
	

	while (current_seq < seq_num)
	{
		file_seq_list >> sSequence;
		file_mask_list >> sMask;
		std::cout << "\n" << sSequence << std::endl;

		bs::CaptureYUV mask(sMask, width, height, chromaSubsampling, 1);
		if (!mask.isOpened()) return EXIT_FAILURE;

		bs::CaptureYUV forwardVideo(sSequence, width, height, chromaSubsampling, 1);
		if (!forwardVideo.isOpened()) return EXIT_FAILURE;

		std::cout << "forward: " << std::endl;
		bs::MarkerTracker forwardTracker(&forwardVideo, &mask);
		forwardTracker.start();
		forward_points.clear();
		forward_points = forwardTracker.getPoints();

		bs::CaptureYUV reverseVideo(sSequence, width, height, chromaSubsampling, -1);
		if (!forwardVideo.isOpened()) return 0;

		std::cout << std::endl << "reverse: " << std::endl;
		bs::MarkerTracker reverseTracker(&reverseVideo, &mask);
		reverseTracker.start();
		reverse_points.clear();
		reverse_points = reverseTracker.getPoints();
		std::reverse(reverse_points.begin(), reverse_points.end());

		//analyze points

		final_points.clear();
		final_points.reserve(forward_points.size());

		for (size_t i = 0; i < forward_points.size(); i++)
		{
			if (forward_points[i] == NOT_DETECTED || reverse_points[i] == NOT_DETECTED || distance(forward_points[i], reverse_points[i]) > 1)
				final_points.emplace_back(NOT_DETECTED);
			else
				final_points.push_back(forward_points[i]);

		}

		//filter single positions
		for (size_t i = 0; i < final_points.size(); i++)
		{
			if (i == final_points.size() - 2
				&& final_points[i] == NOT_DETECTED
				&& final_points[i + 1] != NOT_DETECTED)
			{
				final_points[i + 1] = NOT_DETECTED;
			}

			if (i <= final_points.size() - 3
				&& final_points[i] == NOT_DETECTED
				&& final_points[i + 1] != NOT_DETECTED
				&& final_points[i + 2] == NOT_DETECTED)
			{
				final_points[i + 1] = NOT_DETECTED;
			}
		}

		for (size_t i = 0; i < frame_num; i++)
		{
			points[i][current_seq] = final_points[i];
		}
		
		current_seq++;
		system("cls");
	}

	std::ofstream file_final_pos(sSequence + "\\..\\..\\points.txt", std::ios::out);

	//save to file

	for (size_t i = 0; i < frame_num; i++)
	{
		for (size_t j = 0; j < seq_num; j++)
		{
			file_final_pos << std::setprecision(3) << std::fixed;
			if (points[i][j] == NOT_DETECTED)
				file_final_pos << points[i][j].x << "\t" << points[i][j].y << "\t" << 0 << "\t";
			else
				file_final_pos << points[i][j].x << "\t" << points[i][j].y << "\t" << 1 << "\t";
		}
		file_final_pos << "\n";
	}

	std::cout << "POINTS SAVED\n";
	file_final_pos.close();
	return 0;
}
