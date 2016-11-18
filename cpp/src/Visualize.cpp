#include "Visualize.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;

// Displays final stage heatmaps overlayed on original image
void Visualize(const cv::Mat &testImage, const std::vector<cv::Mat> &heatmaps);

// Displays points of highest probability from final stage heatmap overlayed on original image
void DisplayLimbPoints(const cv::Mat &testImage, const std::vector<cv::Point> &limbPoints) {
  Mat testImageCopy(testImage);

  for (Point limbPoint : limbPoints) {
    circle(testImageCopy, limbPoint, 2, Scalar(0, 255, 0), 4);
  }

  imshow("Limbs", testImageCopy);
  waitKey(0);
}

