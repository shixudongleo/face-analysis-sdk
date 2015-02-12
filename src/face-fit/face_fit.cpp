/*
  author: shixudong
  date:   09/02/2015
  about:  wrapper for face benchmark generator
*/

#include "utils/helpers.hpp"
#include "utils/command-line-arguments.hpp"
#include "tracker/FaceTracker.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

// Wrapper for most external modules
#include <boost/python.hpp>

// np_opencv_converter
#include "face-fit/conversion.h"

using namespace FACETRACKER;

namespace bp = boost::python;

struct Configuration
{
  double wait_time;
  std::string model_pathname;
  std::string params_pathname;
  int tracking_threshold;
  std::string window_title;
  bool verbose;

  int circle_radius;
  int circle_thickness;
  int circle_linetype;
  int circle_shift;
};

class FaceLandmarkDetector
{
public:
	FaceLandmarkDetector()
	{
	  cfg_.wait_time = 0;
	  cfg_.model_pathname = DefaultFaceTrackerModelPathname();
	  cfg_.params_pathname = DefaultFaceTrackerParamsPathname();
	  cfg_.tracking_threshold = 5;
	  cfg_.window_title = "CSIRO Face Fit";
	  cfg_.verbose = false;
	  cfg_.circle_radius = 2;
	  cfg_.circle_thickness = 1;
	  cfg_.circle_linetype = 8;
	  cfg_.circle_shift = 0;

	  tracker_ = LoadFaceTracker(cfg_.model_pathname.c_str());
	  tracker_params_  = LoadFaceTrackerParams(cfg_.params_pathname.c_str());
	}

	~FaceLandmarkDetector()
	{
	  delete tracker_;
	  delete tracker_params_;
	}

	PyObject*
	get_face_landmarks_image_mode(PyObject * in)
	{  
	  //ndarray cv::Mat conversion
	  NDArrayConverter cvt;
	  cv::Mat image = { cvt.toMat(in) };
	  cv::Mat_<uint8_t> gray_image;

	  assert(tracker_);
      assert(tracker_params_);
	   
      if (image.type() == cv::DataType<cv::Vec<uint8_t,3> >::type)
        cv::cvtColor(image, gray_image, CV_BGR2GRAY);
      else 
      	gray_image = image;

      int result = tracker_->NewFrame(gray_image, tracker_params_);

      std::vector<cv::Point_<double> > shape;

      if (result >= cfg_.tracking_threshold) {
        shape = tracker_->getShape();
        tracker_->Reset();
      } 

	  cv::Mat landmarks;
	  if (shape.size() > 0) {
	      landmarks = cv::Mat(shape.size(), 2, CV_32F);

	      for (int i = 0; i < shape.size(); i++) {
	        landmarks.at<float>(i, 0) = shape[i].x;
	        landmarks.at<float>(i, 1) = shape[i].y;
	      }
	  } else {
	      landmarks = cv::Mat(0, 0, CV_32F);
	  } 

	  // cv::Mat to ndarray
	  return cvt.toNDArray(landmarks);
	}

	PyObject*
	get_face_landmarks_video_mode(PyObject * in)
	{  
	  //ndarray cv::Mat conversion
	  NDArrayConverter cvt;
	  cv::Mat image = { cvt.toMat(in) };
	  cv::Mat_<uint8_t> gray_image;

	  assert(tracker_);
      assert(tracker_params_);
	   
      if (image.type() == cv::DataType<cv::Vec<uint8_t,3> >::type)
        cv::cvtColor(image, gray_image, CV_BGR2GRAY);
      else 
      	gray_image = image;

      int result = tracker_->Track(gray_image, tracker_params_);

      std::vector<cv::Point_<double> > shape;

      if (result >= cfg_.tracking_threshold) {
        shape = tracker_->getShape();
      } else {
        tracker_->Reset();
      }

	  cv::Mat landmarks;
	  if (shape.size() > 0) {
	      landmarks = cv::Mat(shape.size(), 2, CV_32F);

	      for (int i = 0; i < shape.size(); i++) {
	        landmarks.at<float>(i, 0) = shape[i].x;
	        landmarks.at<float>(i, 1) = shape[i].y;
	      }
	  } else {
	      landmarks = cv::Mat(0, 0, CV_32F);
	  } 

	  // cv::Mat to ndarray
	  return cvt.toNDArray(landmarks);
	}	

private: 
	Configuration cfg_;
	FaceTracker * tracker_;
	FaceTrackerParams *tracker_params_;

};



static void init()
{
    Py_Initialize();
    import_array();
}

BOOST_PYTHON_MODULE(face_fit)
{
    init();
    bp::class_<FaceLandmarkDetector>("FaceLandmarkDetector")
    		.def("get_face_landmarks_image_mode", &FaceLandmarkDetector::get_face_landmarks_image_mode)
    		.def("get_face_landmarks_video_mode", &FaceLandmarkDetector::get_face_landmarks_video_mode);
}

