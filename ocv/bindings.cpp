#include "bindings.h"
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <iostream>
#include <cstdlib>

#define DEBUG_LOG(msg) std::cout << "DEBUG: " << msg << std::endl

// --- Mat ---
Mat new_mat() { return new cv::Mat(); }

void mat_release(Mat mat) {
  if (mat) {
    delete static_cast<cv::Mat *>(mat);
  }
}

bool mat_isempty(Mat mat) {
  return static_cast<cv::Mat *>(mat)->empty();
}

int mat_rows(Mat mat) { return static_cast<cv::Mat *>(mat)->rows; }

int mat_cols(Mat mat) { return static_cast<cv::Mat *>(mat)->cols; }

int mat_channels(Mat mat) {
  return static_cast<cv::Mat *>(mat)->channels();
}

float *mat_data_f32(Mat mat) {
  cv::Mat *m = static_cast<cv::Mat *>(mat);
  // Ensure the Mat is of type CV_32F
  if (m->type() != CV_32F) {
    return nullptr;
  }
  return reinterpret_cast<float *>(m->data);
}

// --- Core / HighGUI ---

Mat image_read(const char *file, int flags) {
  cv::Mat image = cv::imread(file, flags);
  if (image.empty()) {
    return nullptr;
  }
  return new cv::Mat(image);
}

bool image_write(const char *filename, Mat img) {
  return cv::imwrite(filename, *static_cast<cv::Mat *>(img));
}

void named_window(const char *name) { cv::namedWindow(name); }

void image_show(const char *name, Mat img) {
  cv::imshow(name, *static_cast<cv::Mat *>(img));
}

int wait_key(int delay) { return cv::waitKey(delay); }

void destroy_window(const char *name) { cv::destroyWindow(name); }

void free_mem(void *ptr) { 
    if (ptr) {
        free(ptr);
    }
}


// --- VideoCapture ---

VideoCapture new_videocapture() { return new cv::VideoCapture(); }

void videocapture_release(VideoCapture cap) {
  static_cast<cv::VideoCapture *>(cap)->release();
  delete static_cast<cv::VideoCapture *>(cap);
}

bool videocapture_open(VideoCapture cap, int device_id, int api_id) {
  return static_cast<cv::VideoCapture *>(cap)->open(device_id, api_id);
}

bool videocapture_isopened(VideoCapture cap) {
  return static_cast<cv::VideoCapture *>(cap)->isOpened();
}

bool videocapture_read(VideoCapture cap, Mat frame) {
  return static_cast<cv::VideoCapture *>(cap)->read(
      *static_cast<cv::Mat *>(frame));
}

// --- imgproc Module ---
void cvt_color(Mat src, Mat dst, int code) {
  cv::cvtColor(*static_cast<cv::Mat *>(src), *static_cast<cv::Mat *>(dst), code);
}

void gaussian_blur(Mat src, Mat dst, int ksize_width, int ksize_height,
                      double sigmaX) {
  cv::GaussianBlur(*static_cast<cv::Mat *>(src),
                   *static_cast<cv::Mat *>(dst),
                   cv::Size(ksize_width, ksize_height), sigmaX);
}

void canny(Mat image, Mat edges, double threshold1, double threshold2) {
  cv::Canny(*static_cast<cv::Mat *>(image), *static_cast<cv::Mat *>(edges),
              threshold1, threshold2);
}

void rectangle(Mat img, Rect rect, int b, int g, int r, int thickness) {
  cv::rectangle(*static_cast<cv::Mat *>(img),
                cv::Rect(rect.x, rect.y, rect.width, rect.height),
                cv::Scalar(b, g, r), thickness);
}

void put_text(Mat img, const char *text, int x, int y, int fontFace, double fontScale, int b, int g, int r, int thickness) {
  cv::putText(*static_cast<cv::Mat *>(img), text, cv::Point(x, y), fontFace,
              fontScale, cv::Scalar(b, g, r), thickness);
}

void in_range(Mat src, Scalar lowerb, Scalar upperb, Mat dst) {
    cv::inRange(*static_cast<cv::Mat*>(src), cv::Scalar(lowerb.val1, lowerb.val2, lowerb.val3), cv::Scalar(upperb.val1, upperb.val2, upperb.val3), *static_cast<cv::Mat*>(dst));
}

// Erode and Dilate with a default 3x3 kernel
void erode(Mat src, Mat dst) {
    cv::erode(*static_cast<cv::Mat*>(src), *static_cast<cv::Mat*>(dst), cv::Mat());
}

void dilate(Mat src, Mat dst) {
    cv::dilate(*static_cast<cv::Mat*>(src), *static_cast<cv::Mat*>(dst), cv::Mat());
}

Contours* find_contours(Mat image) {
    std::vector<std::vector<cv::Point>> contours_vec;
    cv::findContours(*static_cast<cv::Mat*>(image), contours_vec, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    if (contours_vec.empty()) {
        return nullptr;
    }

    Contours* c_contours = new Contours;
    c_contours->count = contours_vec.size();
    c_contours->contours = new Contour[c_contours->count];

    for (size_t i = 0; i < contours_vec.size(); ++i) {
        c_contours->contours[i].count = contours_vec[i].size();
        c_contours->contours[i].points = new Point[contours_vec[i].size()];
        for (size_t j = 0; j < contours_vec[i].size(); ++j) {
            c_contours->contours[i].points[j] = {contours_vec[i][j].x, contours_vec[i][j].y};
        }
    }
    return c_contours;
}

void free_contours(Contours* contours) {
    if (!contours) return;
    for (int i = 0; i < contours->count; ++i) {
        delete[] contours->contours[i].points;
    }
    delete[] contours->contours;
    delete contours;
}

double contour_area(Contour* contour) {
    std::vector<cv::Point> vec;
    for(int i = 0; i < contour->count; ++i) {
        vec.push_back(cv::Point(contour->points[i].x, contour->points[i].y));
    }
    return cv::contourArea(vec);
}

RotatedRect min_area_rect(Contour* contour) {
    std::vector<cv::Point> vec;
    for(int i = 0; i < contour->count; ++i) {
        vec.push_back(cv::Point(contour->points[i].x, contour->points[i].y));
    }
    cv::RotatedRect rect = cv::minAreaRect(vec);
    return {
        {rect.center.x, rect.center.y},
        {rect.size.width, rect.size.height},
        rect.angle
    };
}

Point* box_points(RotatedRect rect) {
    cv::RotatedRect rotated(
        cv::Point2f(rect.center.x, rect.center.y),
        cv::Size2f(rect.size.width, rect.size.height),
        rect.angle
    );
    cv::Point2f points[4];
    rotated.points(points);
    Point* c_points = new Point[4];
    for (int i = 0; i < 4; ++i) {
        c_points[i] = {(int)points[i].x, (int)points[i].y};
    }
    return c_points;
}

void free_points(void* points) {
    delete[] (Point*)points;
}

void draw_contours(Mat image, Contours* contours, int contour_idx, Scalar color, int thickness) {
    std::vector<std::vector<cv::Point>> contours_vec;
    contours_vec.resize(contours->count);
    for (int i = 0; i < contours->count; ++i) {
        for(int j = 0; j < contours->contours[i].count; ++j) {
            contours_vec[i].push_back(cv::Point(contours->contours[i].points[j].x, contours->contours[i].points[j].y));
        }
    }
    cv::drawContours(*static_cast<cv::Mat*>(image), contours_vec, contour_idx, cv::Scalar(color.val1, color.val2, color.val3, color.val4), thickness);
}

// Wrapper to avoid needing a callback in Odin for this simple case.
void create_trackbar(const char* trackbar_name, const char* window_name, int* value, int count) {
    // This is the modern, safe version of createTrackbar.
    // 1. We pass 'nullptr' instead of the direct 'value' pointer.
    // 2. We provide a callback function (as a C++ lambda).
    // 3. The 'value' pointer from the function arguments is "captured" by the lambda.
    // 4. When the user moves the slider, OpenCV calls this lambda with the new position.
    // 5. The lambda then writes the new position into the original pointer.
    cv::createTrackbar(trackbar_name, window_name, nullptr, count, 
        [](int pos, void* userdata) {
            // Cast the userdata back to the integer pointer and update its value.
            *static_cast<int*>(userdata) = pos;
        }, 
        // Pass the original 'value' pointer as the userdata.
        value 
    );
}

// --- objdetect Module ---

CascadeClassifier cascade_classifier_new() {
  return new cv::CascadeClassifier();
}

void cascade_classifier_release(CascadeClassifier cc) {
  delete static_cast<cv::CascadeClassifier *>(cc);
}

bool cascade_classifier_load(CascadeClassifier cc, const char *filename) {
  return static_cast<cv::CascadeClassifier *>(cc)->load(filename);
}

Rects* cascade_classifier_detect_multi_scale(CascadeClassifier cc, Mat image) {
  std::vector<cv::Rect> detected_objects;
  static_cast<cv::CascadeClassifier *>(cc)->detectMultiScale(
      *static_cast<cv::Mat *>(image), detected_objects);

  if (detected_objects.empty()) {
    return nullptr;
  }

  Rects *results = new Rects;
  results->count = static_cast<int>(detected_objects.size());
  results->rects = new Rect[results->count];

  for (size_t i = 0; i < detected_objects.size(); ++i) {
    results->rects[i] = {detected_objects[i].x, detected_objects[i].y,
                         detected_objects[i].width,
                         detected_objects[i].height};
  }
  return results;
}

void free_rects(Rects* rects) {
    if (rects) {
        delete[] rects->rects;
        delete rects;
    }
}

// --- dnn Module ---
std::vector<std::string> get_output_layer_names(cv::dnn::Net& net) {
    std::vector<std::string> layer_names = net.getLayerNames();
    // Get the indices of the output layers, i.e. the layers with unconnected outputs
    std::vector<int> out_layers = net.getUnconnectedOutLayers();
    std::vector<std::string> out_layer_names;
    out_layer_names.resize(out_layers.size());
    for (size_t i = 0; i < out_layers.size(); ++i) {
        out_layer_names[i] = layer_names[out_layers[i] - 1];
    }
    return out_layer_names;
}

Net dnn_read_net(const char *model, const char *config) {
  return new cv::dnn::Net(cv::dnn::readNet(model, config));
}

void net_release(Net net) { delete static_cast<cv::dnn::Net *>(net); }

// Helper function (not exported) to get output layer names
std::vector<std::string> get_output_layer_names(const cv::dnn::Net& net) {
    std::vector<std::string> names;
    if (names.empty()) {
        std::vector<int> out_layers = net.getUnconnectedOutLayers();
        std::vector<std::string> layers_names = net.getLayerNames();
        names.resize(out_layers.size());
        for (size_t i = 0; i < out_layers.size(); ++i) {
            names[i] = layers_names[out_layers[i] - 1];
        }
    }
    return names;
}

// --- REFACTORED ALL-IN-ONE DNN OBJECT DETECTION FUNCTION ---
// It no longer allocates memory. It fills a vector provided by the caller (Odin).
Detections* dnn_detect_objects(Net net, Mat image, float conf_threshold, float nms_threshold) {
    // Cast the void* handles back to their C++ types
    cv::dnn::Net* net_ptr = static_cast<cv::dnn::Net*>(net);
    cv::Mat* img_ptr = static_cast<cv::Mat*>(image);

    if (!net_ptr || !img_ptr || img_ptr->empty()) {
        return nullptr;
    }
    
    // --- 1. Pre-process Image ---
    cv::Mat blob;
    cv::dnn::blobFromImage(*img_ptr, blob, 1.0/255.0, cv::Size(416, 416), cv::Scalar(), true, false);

    // --- 2. Set Input and Run Forward Pass ---
    net_ptr->setInput(blob);
    std::vector<cv::Mat> outs;
    net_ptr->forward(outs, get_output_layer_names(*net_ptr));

    // --- 3. Process Outputs ---
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    const float img_width = static_cast<float>(img_ptr->cols);
    const float img_height = static_cast<float>(img_ptr->rows);

    for (const auto& out : outs) {
        const float* data = (const float*)out.data;
        for (int i = 0; i < out.rows; ++i, data += out.cols) {
            cv::Mat scores = out.row(i).colRange(5, out.cols);
            cv::Point class_id_point;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

            if (max_class_score > conf_threshold) {
                int center_x = static_cast<int>(data[0] * img_width);
                int center_y = static_cast<int>(data[1] * img_height);
                int width = static_cast<int>(data[2] * img_width);
                int height = static_cast<int>(data[3] * img_height);
                int left = center_x - width / 2;
                int top = center_y - height / 2;

                class_ids.push_back(class_id_point.x);
                confidences.push_back(static_cast<float>(max_class_score));
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    // --- 4. Perform Non-Maximum Suppression ---
    std::vector<int> indices;
    if (boxes.empty()) {
        return nullptr; // Nothing to do
    }
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

    Detections *results = new Detections;
    results->count = static_cast<int>(boxes.size());
    results->detections = new Detection[results->count];

    for (size_t i = 0; i < boxes.size(); ++i) {
      results->detections[i] = Detection{
            {boxes[i].x, boxes[i].y, boxes[i].width, boxes[i].height},
            class_ids[i],
            confidences[i]
        };
    }
    return results;
}

void free_detections(Detections* detections) {
    if (detections) {
        // First, free the array of detection results
        delete[] detections->detections;
        // Then, free the container struct itself
        delete detections;
    }
}

// --- Kalman Filter Module ---

KalmanFilter kalmanfilter_new() {
  // State: [x, y, w, h, vx, vy, vw, vh] (8 variables, dynamParams)
  // Measurement: [x, y, w, h] (4 variables, measureParams)
  cv::KalmanFilter *kf = new cv::KalmanFilter(8, 4, 0, CV_32F);

  // --- Transition Matrix (A) ---
  // Predicts the next state based on the current one.
  // We use a constant velocity model: x_t = x_{t-1} + vx_{t-1} * dt
  // Assuming dt=1 frame for simplicity.
  cv::setIdentity(kf->transitionMatrix);
  kf->transitionMatrix.at<float>(0, 4) = 1.0f; // x' = x + vx
  kf->transitionMatrix.at<float>(1, 5) = 1.0f; // y' = y + vy
  kf->transitionMatrix.at<float>(2, 6) = 1.0f; // w' = w + vw
  kf->transitionMatrix.at<float>(3, 7) = 1.0f; // h' = h + vh

  // --- Measurement Matrix (H) ---
  // Maps the 8D state space to the 4D measurement space.
  // This was the source of the error.
  kf->measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
  kf->measurementMatrix.at<float>(0, 0) = 1.0f; // Measure x from state x
  kf->measurementMatrix.at<float>(1, 1) = 1.0f; // Measure y from state y
  kf->measurementMatrix.at<float>(2, 2) = 1.0f; // Measure w from state w
  kf->measurementMatrix.at<float>(3, 3) = 1.0f; // Measure h from state h
  
  // --- Noise Covariance Matrices ---
  // Q: Process Noise - uncertainty in our motion model (e.g., face might accelerate)
  cv::setIdentity(kf->processNoiseCov, cv::Scalar::all(1e-2));

  // R: Measurement Noise - uncertainty from the detector (detectors are noisy)
  cv::setIdentity(kf->measurementNoiseCov, cv::Scalar::all(1e-1));

  // P: Initial Error Covariance - how much we trust our initial state.
  cv::setIdentity(kf->errorCovPost, cv::Scalar::all(1));

  return static_cast<KalmanFilter>(kf);
}

void kalmanfilter_set_state_post(KalmanFilter kf, Rect statePost) {
  cv::KalmanFilter *kfPtr = static_cast<cv::KalmanFilter *>(kf);
  kfPtr->statePost.at<float>(0) = static_cast<float>(statePost.x);
  kfPtr->statePost.at<float>(1) = static_cast<float>(statePost.y);
  kfPtr->statePost.at<float>(2) = static_cast<float>(statePost.width);
  kfPtr->statePost.at<float>(3) = static_cast<float>(statePost.height);
  // Initialize velocities to zero
  kfPtr->statePost.at<float>(4) = 0.0f;
  kfPtr->statePost.at<float>(5) = 0.0f;
  kfPtr->statePost.at<float>(6) = 0.0f;
  kfPtr->statePost.at<float>(7) = 0.0f;
}

void kalmanfilter_release(KalmanFilter kf) {
  delete static_cast<cv::KalmanFilter *>(kf);
}

Rect kalmanfilter_predict(KalmanFilter kf) {
  cv::Mat prediction = static_cast<cv::KalmanFilter *>(kf)->predict();
  return Rect{(int)prediction.at<float>(0), (int)prediction.at<float>(1),
                (int)prediction.at<float>(2), (int)prediction.at<float>(3)};
}

Rect kalmanfilter_correct(KalmanFilter kf, Rect measurement) {
  cv::Mat measurementMat =
      (cv::Mat_<float>(4, 1) << measurement.x, measurement.y, measurement.width,
       measurement.height);
  cv::Mat corrected =
      static_cast<cv::KalmanFilter *>(kf)->correct(measurementMat);
  return Rect{(int)corrected.at<float>(0), (int)corrected.at<float>(1),
                (int)corrected.at<float>(2), (int)corrected.at<float>(3)};
}