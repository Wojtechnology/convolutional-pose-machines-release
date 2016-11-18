#pragma once

#include <opencv2/core/core.hpp>

#include <string>
#include <vector>

// Displays final stage heatmaps overlayed on original image
void Visualize(const cv::Mat &testImage, const std::vector<cv::Mat> &heatmaps);

// Displays points of highest probability from final stage heatmap overlayed on original image
void DisplayLimbPoints(const cv::Mat &testImage, const std::vector<cv::Point> &limbPoints);
