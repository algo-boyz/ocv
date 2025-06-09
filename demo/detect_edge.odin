package main

import "core:c"
import "core:fmt"
import "core:os"
import "../ocv"

demo_edge_detection :: proc() {
    capture := ocv.new_videocapture()
    if capture == nil {
        fmt.eprintln("Error: could not create a video capture.")
        os.exit(1)
    }
    defer ocv.videocapture_release(capture)

    device_id: c.int = 0
    ocv.videocapture_open(capture, device_id, .CAP_ANY) 
    if !ocv.videocapture_isopened(capture) {
        fmt.eprintfln("Error: can't open camera stream for device_id=%d.", device_id)
        os.exit(1)
    }

    frame := ocv.new_mat()
    defer ocv.mat_release(frame)
    gray_frame := ocv.new_mat()
    defer ocv.mat_release(gray_frame)
    blurred_frame := ocv.new_mat()
    defer ocv.mat_release(blurred_frame)
    edge_frame := ocv.new_mat()
    defer ocv.mat_release(edge_frame)

    window_name_orig :: "Live Camera"
    window_name_edges :: "Canny Edge Detection"
    ocv.named_window(window_name_orig)
    ocv.named_window(window_name_edges)
    defer ocv.destroy_window(window_name_orig)
    defer ocv.destroy_window(window_name_edges)

    fmt.println(">>> Starting Real-time Edge Detection... Press ESC to exit.")

    for {
        if !ocv.videocapture_read(capture, frame) {
            fmt.eprintln("Error: could not read frame from camera.")
            break
        }
        if ocv.mat_isempty(frame) {
            fmt.eprintln("Error: received an empty frame... exiting.")
            break
        }

        ocv.cvt_color(frame, gray_frame, .BGR2GRAY)
        ocv.gaussian_blur(gray_frame, blurred_frame, 5, 5, 0)
        ocv.canny(blurred_frame, edge_frame, 50, 150)
        
        ocv.image_show(window_name_orig, frame)
        ocv.image_show(window_name_edges, edge_frame)

        if ocv.wait_key(25) == 27 {
            break
        }
    }
}