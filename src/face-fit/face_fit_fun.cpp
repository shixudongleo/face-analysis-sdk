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



PyObject*
get_face_landmarks(PyObject * in)
{
    //ndarray cv::Mat conversion
    NDArrayConverter cvt;
    cv::Mat gray_image { cvt.toMat(in) };
    
    CommandLineArgument<std::string> landmarks_argument;
    
    Configuration cfg;
    cfg.wait_time = 0;
    cfg.model_pathname = DefaultFaceTrackerModelPathname();
    cfg.params_pathname = DefaultFaceTrackerParamsPathname();
    cfg.tracking_threshold = 5;
    cfg.window_title = "CSIRO Face Fit";
    cfg.verbose = false;
    cfg.circle_radius = 2;
    cfg.circle_thickness = 1;
    cfg.circle_linetype = 8;
    cfg.circle_shift = 0;
    
    
    FaceTracker * tracker = LoadFaceTracker(cfg.model_pathname.c_str());
    FaceTrackerParams *tracker_params  = LoadFaceTrackerParams(cfg.params_pathname.c_str());
    
    
    int result = tracker->NewFrame(gray_image, tracker_params);
    
    std::vector<cv::Point_<double> > shape;
    
    if (result >= cfg.tracking_threshold)
        shape = tracker->getShape();
    else {
        tracker->Reset();
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
    
    delete tracker;
    delete tracker_params;
    
    // cv::Mat to ndarray
    return cvt.toNDArray(landmarks);
}


static void init()
{
    Py_Initialize();
    import_array();
}

BOOST_PYTHON_MODULE(face_fit_fun)
{
    init();
    bp::def("get_face_landmarks", get_face_landmarks);
}