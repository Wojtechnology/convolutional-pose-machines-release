#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "PoseEstimator.hpp"
#include "Visualize.hpp"

using namespace cv;

int main(int argc, char **argv) {
  if (argc < 4) {
    printf("usage: %s model_file trained_file test_file\n", argv[0]);
  }
  std::string modelFile(argv[1]);
  std::string trainedFile(argv[2]);
  std::string testFile(argv[3]);

  Mat testImage;
  testImage = imread(testFile, CV_LOAD_IMAGE_COLOR);
  PoseEstimator poseEstimator(modelFile, trainedFile);
  std::vector<Point> limbPoints = poseEstimator.detectLimbs(testImage);

  DisplayLimbPoints(testImage, limbPoints);
}
