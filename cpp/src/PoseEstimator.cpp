#include "PoseEstimator.hpp"

#include <caffe/caffe.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;
using namespace cv;

void WrapInputLayer(std::vector<Mat>* input_channels, const Net<float>& net) {
  Blob<float>* input_layer = net.input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Preprocess(const Mat& img, std::vector<Mat>* input_channels,
                int num_channels, const Size& input_geometry) {
  Mat sample;
  if (img.channels() == 3) {
    cvtColor(img, sample, COLOR_RGB2BGRA);
  } else if (img.channels() == 4) {
    cvtColor(img, sample, COLOR_RGBA2BGRA);
    // TODO(wojtek): Find out why this is here
    // sample = img;
  } else {
    printf("Image not supported");
    return;
  }

  Mat sample_resized;
  if (sample.size() != input_geometry) {
    resize(sample, sample_resized, input_geometry);
  } else {
    sample_resized = sample;
  }

  Mat normalized = Mat::zeros(sample_resized.size().height,
                              sample_resized.size().width, CV_32FC4);
  for (int i = 0; i < sample_resized.size().width; ++i) {
    for (int j = 0; j < sample_resized.size().height; ++j) {
      normalized.at<Vec4f>(i, j)[0] =
          float(sample_resized.at<Vec4b>(i, j)[0]) / 255 - 0.5;
      normalized.at<Vec4f>(i, j)[1] =
          float(sample_resized.at<Vec4b>(i, j)[1]) / 255 - 0.5;
      normalized.at<Vec4f>(i, j)[2] =
          float(sample_resized.at<Vec4b>(i, j)[2]) / 255 - 0.5;
      normalized.at<Vec4f>(i, j)[3] = 0.5;
    }
  }

  split(normalized, *input_channels);
}

PoseEstimator::PoseEstimator(const std::string &modelFile, const std::string &trainedFile) :
    modelFile_(modelFile), trainedFile_(trainedFile) {
  // Initialize Caffe.
  Caffe::set_mode(Caffe::CPU);
  net_ = std::unique_ptr<Net<float>>(new Net<float>(modelFile, TEST));
  net_->CopyTrainedLayersFrom(trainedFile);
  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  input_geometry_ = Size(input_layer->width(), input_layer->height());
  input_layer->Reshape(1, num_channels_, input_geometry_.height,
                       input_geometry_.width);
  net_->Reshape();
}

std::vector<Mat> PoseEstimator::predictHeatmaps(const Mat &humanRect) {
  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels, *net_);
  Preprocess(humanRect, &input_channels, num_channels_, input_geometry_);

  net_->Forward();
  Blob<float>* output_layer = net_->output_blobs()[0];
  float* channel = output_layer->mutable_cpu_data();

  std::vector<Mat> heatmaps;
  for (int i = 0; i < output_layer->channels(); ++i) {
    Mat class_heatmap(output_layer->height(), output_layer->width(), CV_32F,
                      channel);
    resize(class_heatmap, class_heatmap, humanRect.size());

    // exaggerate heatmap (not sure if necessary)
    // double min, max;
    // cv::minMaxLoc(class_heatmap, &min, &max);
    // for (int i = 0; i < class_heatmap.size().width; ++i) {
    //   for (int j = 0; j < class_heatmap.size().height; ++j) {
    //     class_heatmap.at<float>(i, j) =
    //         (class_heatmap.at<float>(i, j) - min) / (max - min);
    //   }
    // }

    heatmaps.push_back(class_heatmap);
    channel += output_layer->width() * output_layer->height();
  }

  return heatmaps;
}

std::vector<Point> PoseEstimator::detectLimbs(const Mat &humanRect) {
  std::vector<Mat> heatmaps = predictHeatmaps(humanRect);
  std::vector<Point> limbPoints;
  for (int i = 0; i < heatmaps.size(); ++i) {
    Point max_loc;
    cv::minMaxLoc(heatmaps[i], NULL, NULL, NULL, &max_loc);
    limbPoints.push_back(max_loc);
  }
  return limbPoints;
}
