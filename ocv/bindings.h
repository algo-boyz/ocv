#ifndef IMPROC_H
#define IMPROC_H
#include <stdbool.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>

typedef void *Mat;
typedef void *VideoCapture;
typedef void *CascadeClassifier;
typedef void *Net;
typedef void *KalmanFilter;

struct Point { int x, y; };
struct Point2f { float x, y; };
struct Size2f { float width, height; };
struct Scalar { double val1, val2, val3, val4; };

struct RotatedRect {
    Point2f center;
    Size2f size;
    float angle;
};

struct Contour {
    Point* points;
    int count;
};

struct Contours {
    Contour* contours;
    int count;
};

// C-compatible struct to hold a rectangle
// This struct is used for bounding boxes in object detection and tracking.
typedef struct {
  int x;
  int y;
  int width;
  int height;
} Rect;

// C-compatible struct to hold an array of rectangles for detection results
typedef struct {
    Rect* rects;
    int count;
} Rects;

// C-compatible detection struct for DNN object detection
// This struct holds the bounding box, class ID, and confidence score for each detected object.
struct Detection {
    Rect box;
    int class_id;
    float confidence;
};

// C-compatible struct to hold an array of detections
// This struct is used to return multiple detections from the DNN module.
typedef struct {
    Detection* detections;
    int count;
} Detections;

#ifdef __cplusplus
extern "C" {
#endif

//! Imread flags
enum ImageReadModes {
  IMREAD_UNCHANGED =
      -1, //!< If set, return the loaded image as is (with alpha channel,
          //!< otherwise it gets cropped). Ignore EXIF orientation.
  IMREAD_GRAYSCALE = 0, //!< If set, always convert image to the single channel
                        //!< grayscale image (codec internal conversion).
  IMREAD_COLOR =
      1, //!< If set, always convert image to the 3 channel BGR color image.
  IMREAD_ANYDEPTH =
      2, //!< If set, return 16-bit/32-bit image when the input has the
         //!< corresponding depth, otherwise convert it to 8-bit.
  IMREAD_ANYCOLOR =
      4, //!< If set, the image is read in any possible color format.
  IMREAD_LOAD_GDAL = 8, //!< If set, use the gdal driver for loading the image.
  IMREAD_REDUCED_GRAYSCALE_2 =
      16, //!< If set, always convert image to the single channel grayscale
          //!< image and the image size reduced 1/2.
  IMREAD_REDUCED_COLOR_2 =
      17, //!< If set, always convert image to the 3 channel BGR color image
          //!< and the image size reduced 1/2.
  IMREAD_REDUCED_GRAYSCALE_4 =
      32, //!< If set, always convert image to the single channel grayscale
          //!< image and the image size reduced 1/4.
  IMREAD_REDUCED_COLOR_4 =
      33, //!< If set, always convert image to the 3 channel BGR color image
          //!< and the image size reduced 1/4.
  IMREAD_REDUCED_GRAYSCALE_8 =
      64, //!< If set, always convert image to the single channel grayscale
          //!< image and the image size reduced 1/8.
  IMREAD_REDUCED_COLOR_8 =
      65, //!< If set, always convert image to the 3 channel BGR color image
          //!< and the image size reduced 1/8.
  IMREAD_IGNORE_ORIENTATION = 128 //!< If set, do not rotate the image
                                  //!< according to EXIF's orientation flag.
};

// --- Mat ---
Mat new_mat();
void mat_release(Mat mat);
bool mat_isempty(Mat mat);
int mat_rows(Mat mat);
int mat_cols(Mat mat);
int mat_channels(Mat mat);
float* mat_data_f32(Mat mat);

// --- Core / HighGUI ---
Mat image_read(const char *file, int flags);
bool image_write(const char *filename, Mat img);
void named_window(const char *name);
void image_show(const char *name, Mat img);
int wait_key(int delay);
void destroy_window(const char *name);
void free_mem(void* ptr);

// --- VideoCapture ---
VideoCapture new_videocapture();
void videocapture_release(VideoCapture cap);
bool videocapture_open(VideoCapture cap, int device_id, int api_id);
bool videocapture_isopened(VideoCapture cap);
bool videocapture_read(VideoCapture cap, Mat frame);

// --- imgproc Module ---
void cvt_color(Mat src, Mat dst, int code);
void gaussian_blur(Mat src, Mat dst, int ksize_width, int ksize_height, double sigmaX);
void canny(Mat image, Mat edges, double threshold1, double threshold2);
void rectangle(Mat img, Rect rect, int b, int g, int r, int thickness);
void put_text(Mat img, const char* text, int x, int y, int fontFace, double fontScale, int b, int g, int r, int thickness);
void in_range(Mat src, Scalar lowerb, Scalar upperb, Mat dst);
void erode(Mat src, Mat dst);
void dilate(Mat src, Mat dst);

// --- Contours ---
// FIXED SIGNATURES
Contours* find_contours(Mat image);
void free_contours(Contours* contours);
double contour_area(Contour* contour);
RotatedRect min_area_rect(Contour* contour);
Point* box_points(RotatedRect rect);
void free_points(void* points);
void draw_contours(Mat image, Contours* contours, int contour_idx, Scalar color, int thickness);
// Wrapper to avoid needing a callback in Odin for this simple case.
void create_trackbar(const char* trackbar_name, const char* window_name, int* value, int count);

// --- objdetect Module ---
CascadeClassifier cascade_classifier_new();
void cascade_classifier_release(CascadeClassifier cc);
bool cascade_classifier_load(CascadeClassifier cc, const char* filename);
Rects* cascade_classifier_detect_multi_scale(CascadeClassifier cc, Mat image);
void free_rects(Rects* rects);

// --- dnn Module ---
Net dnn_read_net(const char* model, const char* config);
void net_release(Net net);
Detections* dnn_detect_objects(Net net, Mat image, float conf_threshold, float nms_threshold);
void free_detections(Detections* detections);

// --- Kalman Filter Module (FIXED SIGNATURES) ---
KalmanFilter kalmanfilter_new();
void kalmanfilter_set_state_post(KalmanFilter kf, Rect statePost);
void kalmanfilter_release(KalmanFilter kf);
Rect kalmanfilter_predict(KalmanFilter kf);
Rect kalmanfilter_correct(KalmanFilter kf, Rect measurement);

#ifdef __cplusplus
}
#endif

#endif