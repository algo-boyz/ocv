package ocv

import "core:c"

// when ODIN_OS == .Linux do foreign import cv "imageprocessing.a"
when ODIN_OS == .Linux do foreign import cv "../build/ocv.so"
else when ODIN_OS == .Darwin do foreign import cv "../build/ocv.dylib"

Mat :: distinct rawptr
VideoCapture :: distinct rawptr
CascadeClassifier :: distinct rawptr
KalmanFilter :: distinct rawptr
Net :: distinct rawptr

// Imgproc module structures
Point :: struct {
    x, y: c.int,
}

Point2f :: struct {
    x, y: c.float,
}

Size2f :: struct {
    width, height: c.float,
}

Scalar :: struct {
    val1, val2, val3, val4: c.double,
}

RotatedRect :: struct {
    center: Point2f,
    size:   Size2f,
    angle:  c.float,
}

Contour :: struct {
    points: ^Point,
    count:  c.int,
}

Contours :: struct {
    contours: ^Contour,
    count:    c.int,
}

// Face detection structures
Rect :: struct {
    x, y, width, height: c.int,
}

Rects :: struct {
    rects: ^Rect,
    count: c.int,
}

// Dnn module structures
Detections :: struct {
    detections: ^Detection,
    count: c.int,
}

Detection :: struct {
    box:        Rect,
    classId:    c.int,
    confidence: c.float,
}

ImageReadModes :: enum c.int {
	IMREAD_UNCHANGED           = -1, //!< If set, return the loaded image as is (with alpha channel,
	//!< otherwise it gets cropped). Ignore EXIF orientation.
	IMREAD_GRAYSCALE           = 0, //!< If set, always convert image to the single channel
	//!< grayscale image (codec internal conversion).
	IMREAD_COLOR               = 1, //!< If set, always convert image to the 3 channel BGR color image.
	IMREAD_ANYDEPTH            = 2, //!< If set, return 16-bit/32-bit image when the input has the
	//!< corresponding depth, otherwise convert it to 8-bit.
	IMREAD_ANYCOLOR            = 4, //!< If set, the image is read in any possible color format.
	IMREAD_LOAD_GDAL           = 8, //!< If set, use the gdal driver for loading the image.
	IMREAD_REDUCED_GRAYSCALE_2 = 16, //!< If set, always convert image to the single channel grayscale
	//!< image and the image size reduced 1/2.
	IMREAD_REDUCED_COLOR_2     = 17, //!< If set, always convert image to the 3 channel BGR color image
	//!< and the image size reduced 1/2.
	IMREAD_REDUCED_GRAYSCALE_4 = 32, //!< If set, always convert image to the single channel grayscale
	//!< image and the image size reduced 1/4.
	IMREAD_REDUCED_COLOR_4     = 33, //!< If set, always convert image to the 3 channel BGR color image
	//!< and the image size reduced 1/4.
	IMREAD_REDUCED_GRAYSCALE_8 = 64, //!< If set, always convert image to the single channel grayscale
	//!< image and the image size reduced 1/8.
	IMREAD_REDUCED_COLOR_8     = 65, //!< If set, always convert image to the 3 channel BGR color image
	//!< and the image size reduced 1/8.
	IMREAD_IGNORE_ORIENTATION  = 128, //!< If set, do not rotate the image
	//!< according to EXIF's orientation flag.
}

VideoCaptureAPI :: enum c.int {
	CAP_ANY           = 0, //!< Auto detect == 0
	CAP_VFW           = 200, //!< Video For Windows (obsolete, removed)
	CAP_V4L           = 200, //!< V4L/V4L2 capturing support
	CAP_V4L2          = CAP_V4L, //!< Same as CAP_V4L
	CAP_FIREWIRE      = 300, //!< IEEE 1394 drivers
	CAP_FIREWARE      = CAP_FIREWIRE, //!< Same value as CAP_FIREWIRE
	CAP_IEEE1394      = CAP_FIREWIRE, //!< Same value as CAP_FIREWIRE
	CAP_DC1394        = CAP_FIREWIRE, //!< Same value as CAP_FIREWIRE
	CAP_CMU1394       = CAP_FIREWIRE, //!< Same value as CAP_FIREWIRE
	CAP_QT            = 500, //!< QuickTime (obsolete, removed)
	CAP_UNICAP        = 600, //!< Unicap drivers (obsolete, removed)
	CAP_DSHOW         = 700, //!< DirectShow (via videoInput)
	CAP_PVAPI         = 800, //!< PvAPI, Prosilica GigE SDK
	CAP_OPENNI        = 900, //!< OpenNI (for Kinect)
	CAP_OPENNI_ASUS   = 910, //!< OpenNI (for Asus Xtion)
	CAP_ANDROID       = 1000, //!< MediaNDK (API Level 21+) and NDK Camera (API level 24+) for Android
	CAP_XIAPI         = 1100, //!< XIMEA Camera API
	CAP_AVFOUNDATION  = 1200, //!< AVFoundation framework for iOS (OS X Lion will have the same API)
	CAP_GIGANETIX     = 1300, //!< Smartek Giganetix GigEVisionSDK
	CAP_MSMF          = 1400, //!< Microsoft Media Foundation (via videoInput). See platform specific notes above.
	CAP_WINRT         = 1410, //!< Microsoft Windows Runtime using Media Foundation
	CAP_INTELPERC     = 1500, //!< RealSense (former Intel Perceptual Computing SDK)
	CAP_REALSENSE     = 1500, //!< Synonym for CAP_INTELPERC
	CAP_OPENNI2       = 1600, //!< OpenNI2 (for Kinect)
	CAP_OPENNI2_ASUS  = 1610, //!< OpenNI2 (for Asus Xtion and Occipital Structure sensors)
	CAP_OPENNI2_ASTRA = 1620, //!< OpenNI2 (for Orbbec Astra)
	CAP_GPHOTO2       = 1700, //!< gPhoto2 connection
	CAP_GSTREAMER     = 1800, //!< GStreamer
	CAP_FFMPEG        = 1900, //!< Open and record video file or stream using the FFMPEG library
	CAP_IMAGES        = 2000, //!< OpenCV Image Sequence (e.g. img_%02d.jpg)
	CAP_ARAVIS        = 2100, //!< Aravis SDK
	CAP_OPENCV_MJPEG  = 2200, //!< Built-in OpenCV MotionJPEG codec
	CAP_INTEL_MFX     = 2300, //!< Intel MediaSDK
	CAP_XINE          = 2400, //!< XINE engine (Linux)
	CAP_UEYE          = 2500, //!< uEye Camera API
	CAP_OBSENSOR      = 2600, //!< For Orbbec 3D-Sensor device/module (Astra+, Femto, Astra2, Gemini2, Gemini2L, Gemini2XL, Femto Mega) attention: Astra2, Gemini2, and Gemini2L cameras currently only support Windows and Linux kernel versions no higher than 4.15, and higher versions of Linux kernel may have exceptions.
}

ColorConversionCodes :: enum c.int {
    BGR2GRAY = 6,
    RGB2GRAY = 7,
    GRAY2BGR = 8,
    BGR2HSV  = 40,
    RGB2HSV  = 41,
}

HersheyFonts :: enum c.int {
    SIMPLEX         = 0,
    PLAIN           = 1,
    DUPLEX          = 2,
    COMPLEX         = 3,
    TRIPLEX         = 4,
    COMPLEX_SMALL   = 5,
    SCRIPT_SIMPLEX  = 6,
    SCRIPT_COMPLEX  = 7,
}

@(default_calling_convention = "c")
foreign cv {
    // --- Mat ---
    new_mat :: proc() -> Mat ---
    mat_release :: proc(mat: Mat) ---
    mat_isempty :: proc(mat: Mat) -> bool ---
    mat_rows :: proc(mat: Mat) -> c.int ---
    mat_cols :: proc(mat: Mat) -> c.int ---
    mat_channels :: proc(mat: Mat) -> c.int ---
    mat_data_f32 :: proc(mat: Mat) -> ^f32 ---

    // --- Core / HighGUI ---
    image_read :: proc(file: cstring, flags: ImageReadModes) -> Mat ---
    image_write :: proc(filename: cstring, img: Mat) -> bool ---
    named_window :: proc(name: cstring) ---
    image_show :: proc(name: cstring, img: Mat) ---
    wait_key :: proc(delay: c.int) -> c.int ---
    destroy_window :: proc(name: cstring) ---
    free_mem :: proc(ptr: rawptr) ---

    // --- VideoCapture ---
    new_videocapture :: proc() -> VideoCapture ---
    videocapture_release :: proc(cap: VideoCapture) ---
    videocapture_open :: proc(cap: VideoCapture, device_id: c.int, api_id: VideoCaptureAPI) -> bool ---
    videocapture_isopened :: proc(cap: VideoCapture) -> bool ---
    videocapture_read :: proc(cap: VideoCapture, frame: Mat) -> bool ---

    // --- imgproc Module ---
    cvt_color :: proc(src: Mat, dst: Mat, code: ColorConversionCodes) ---
    gaussian_blur :: proc(src: Mat, dst: Mat, ksize_width, ksize_height: c.int, sigmaX: c.double) ---
    canny :: proc(image, edges: Mat, threshold1, threshold2: c.double) ---
    rectangle :: proc(img: Mat, rect: Rect, b, g, r, thickness: c.int) ---
    put_text :: proc(img: Mat, text: cstring, x, y: c.int, fontFace: HersheyFonts, fontScale: c.double, b, g, r, thickness: c.int) ---
    in_range :: proc(src: Mat, lowerb, upperb: Scalar, dst: Mat) ---
    erode :: proc(src, dst: Mat) ---
    dilate :: proc(src, dst: Mat) ---
    find_contours :: proc(image: Mat) -> ^Contours ---
    free_contours :: proc(contours: ^Contours) ---
    contour_area :: proc(contour: ^Contour) -> c.double ---
    min_area_rect :: proc(contour: ^Contour) -> RotatedRect ---
    box_points :: proc(rect: RotatedRect) -> ^[4]Point ---
    free_points :: proc(points: rawptr) ---
    draw_contours :: proc(image: Mat, contours: ^Contours, contour_idx: c.int, color: Scalar, thickness: c.int) ---
    create_trackbar :: proc(trackbar_name, window_name: cstring, value: ^c.int, count: c.int) ---

    // --- objdetect Module ---
    cascade_classifier_new :: proc() -> CascadeClassifier ---
    cascade_classifier_release :: proc(cc: CascadeClassifier) ---
    cascade_classifier_load :: proc(cc: CascadeClassifier, filename: cstring) -> bool ---
    cascade_classifier_detect_multi_scale :: proc(cc: CascadeClassifier, image: Mat) -> ^Rects ---
    free_rects :: proc(rects: ^Rects) ---

    // --- dnn Module ---
    dnn_read_net :: proc(model, config: cstring) -> Net ---
    net_release :: proc(net: Net) ---
    dnn_detect_objects :: proc(net: Net, image: Mat, conf_threshold: c.float, nms_threshold: c.float) -> ^Detections ---
    free_detections :: proc(detections: ^Detections) ---

    // --- Kalman Filter procedures ---
    kalmanfilter_new :: proc() -> KalmanFilter ---
    kalmanfilter_set_state_post :: proc(kf: KalmanFilter, state: Rect) ---
    kalmanfilter_release :: proc(kf: KalmanFilter) ---

    // Predicts the next state, returns a Rect with the predicted [x, y, w, h]
    kalmanfilter_predict :: proc(kf: KalmanFilter) -> Rect ---

    // Corrects the state based on a measurement, returns the corrected [x, y, w, h]
    kalmanfilter_correct :: proc(kf: KalmanFilter, measurement: Rect) -> Rect ---
}