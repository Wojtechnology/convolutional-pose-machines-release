#pragma once

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>

class PoseEstimator {
public:
  PoseEstimator(const std::string &modelFile, const std::string &trainedFile);

  // TODO(wojtek): Remove this restriction
  // Human rect must be a square.
  // Returns limbs in order:
  // {'head', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri', 'Rhip',
  // 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'bkg'}
  std::vector<cv::Point> detectLimbs(const cv::Mat &humanRect);

  // Returns heatmaps for all limbs represented by model (for last stage)
  std::vector<cv::Mat> predictHeatmaps(const cv::Mat &humanRect);

private:
  std::string modelFile_;
  std::string trainedFile_;
  std::unique_ptr<caffe::Net<float>> net_;
  cv::Size input_geometry_;
  int num_channels_;
};
