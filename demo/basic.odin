package main

import "core:c"
import "core:fmt"
import "core:os"
import "core:strings"
import "core:slice"
import "../ocv"

demo_read_image :: proc() {
    image_path :: "assets/test.png"
    // `ocv.image_read` returns a pointer to a C++ `Mat` object created with `new`.
    image := ocv.image_read(image_path, ocv.ImageReadModes.IMREAD_COLOR)
    // It MUST be freed with `ocv.mat_release`, which calls `delete`.
    defer ocv.mat_release(image)

    if ocv.mat_isempty(image) {
        fmt.eprintfln("Error: can't find or open an image: %s", image_path)
        os.exit(1)
    }

    window_name :: "Image Demo"
    ocv.named_window(window_name)
    defer ocv.destroy_window(window_name)

    ocv.image_show(window_name, image)
    ocv.wait_key(0)
}

demo_read_camera :: proc() {
    frame := ocv.new_mat()
    // `frame` is a `new cv::Mat*`, so it must be released with `ocv.mat_release`.
    defer ocv.mat_release(frame)

    capture := ocv.new_videocapture()
    // `ocv.videocapture_release` correctly calls `delete` on the capture object.
    // Calling `ocv.free_mem(capture)` afterwards would be a double-free and heap corruption.
    defer ocv.videocapture_release(capture)

    device_id: c.int = 0
    ocv.videocapture_open(capture, device_id, ocv.VideoCaptureAPI.CAP_ANY)

    window_name :: "Camera Video"
    ocv.named_window(window_name)
    defer ocv.destroy_window(window_name)

    if !ocv.videocapture_isopened(capture) {
        fmt.eprintfln(
            "Error: can't open camera stream for device_id=%d and api_id=%d",
            device_id,
            ocv.VideoCaptureAPI.CAP_ANY,
        )
        os.exit(1)
    }

    fmt.println(">>> Reading frames...")
    for {
        ocv.videocapture_read(capture, frame)
        if ocv.mat_isempty(frame) {
            fmt.eprintln("Error: empty frame... exiting")
            break
        }
        ocv.image_show(window_name, frame)
        c := ocv.wait_key(25)
        if c == 27 {    // ESC key pressed
            break
        }
    }
}
