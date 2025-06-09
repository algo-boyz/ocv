package main

import "core:fmt"
import "core:math"
import "core:os"
import "core:slice"
import "core:c"
import "../ocv"

// based on: https://github.com/ariwasch/OpenCV-Distance-Angle-Demo/blob/master/Vision.py

// A struct to hold the state and configuration for vision tracking.
Vision :: struct {
    focal_length:  f64,
    angle:         f64,
    fitted_height: f64,
    fitted_width:  f64,
    known_width:   f64,
    known_height:  f64,
}

// Creates and initializes a new Vision object.
vision_new :: proc(pixel_height, known_distance, known_width, known_height: f64) -> ^Vision {
    vision := new(Vision)
    vision.known_width = known_width
    vision.known_height = known_height
    vision.focal_length = (pixel_height * known_distance) / known_height
    return vision
}

// Deallocates the Vision object.
vision_destroy :: proc(v: ^Vision) {
    free(v)
}

// Calculates the Euclidean distance between two points.
_calculate_distance :: proc(p1, p2: ocv.Point) -> f64 {
    dx := f64(p2.x - p1.x)
    dy := f64(p2.y - p1.y)
    return math.sqrt(dx*dx + dy*dy)
}

// Returns the calculated distance from the camera to the tracked object.
vision_get_distance :: proc(v: ^Vision) -> f64 {
    if v.fitted_height > 0 {
        return ((v.known_height * v.focal_length) / v.fitted_height)
    }
    return 0
}

// Calculates and returns the angle of the object relative to the camera.
vision_get_angle :: proc(v: ^Vision) -> f64 {
    ratio := v.known_width / v.known_height
    w2 := v.fitted_height * ratio
    if v.known_width != 0 && w2 > v.fitted_width {
        v.angle = (1 - (v.fitted_width / w2)) * 90
    } else {
        v.angle = 0
    }
    return v.angle
}

// Processes a video frame to find and analyze a colored object.
// It modifies the input frame by drawing contours.
vision_update_frame :: proc(v: ^Vision, frame: ocv.Mat, hsv_lower, hsv_upper: ocv.Scalar) -> (mask: ocv.Mat) {
    // 1. Create temporary Mats for processing stages
    hsv_frame := ocv.new_mat()
    defer ocv.mat_release(hsv_frame)
    
    mask = ocv.new_mat() // Returned to the caller
    
    eroded_mask := ocv.new_mat()
    defer ocv.mat_release(eroded_mask)

    // 2. Convert frame to HSV and create a color mask
    ocv.cvt_color(frame, hsv_frame, .BGR2HSV)
    ocv.in_range(hsv_frame, hsv_lower, hsv_upper, mask)
    
    // 3. Clean up the mask with erosion and dilation to remove noise
    ocv.erode(mask, eroded_mask)
    ocv.dilate(eroded_mask, mask) // Dilate back into original mask mat

    // 4. Find contours in the cleaned mask
    contours := ocv.find_contours(mask)
    if contours == nil {
        v.fitted_height = 0
        v.fitted_width = 0
        return
    }
    defer ocv.free_contours(contours)

    // 5. Find the largest contour by area
    largest_contour_idx := -1
    max_area: f64 = 0.0
    
    contour_slice := slice.from_ptr(contours.contours, int(contours.count))
    for &contour, i in contour_slice {
        area := ocv.contour_area(&contour)
        if area > max_area {
            max_area = area
            largest_contour_idx = i
        }
    }

    // 6. If a contour was found, process it
    if largest_contour_idx != -1 {
        largest_contour := &contour_slice[largest_contour_idx]
        
        // Get the minimum rotated rectangle
        rect := ocv.min_area_rect(largest_contour)
        box_points_ptr := ocv.box_points(rect)
        if box_points_ptr == nil { return }
        defer ocv.free_points(box_points_ptr)
        
        box_points := box_points_ptr^
        
        // Draw the rotated rectangle contour
        // NOTE: ocv.draw_contours needs a ^Contours struct, so we build a temporary one.
        temp_contour_group := ocv.Contours{contours = largest_contour, count = 1}
        ocv.draw_contours(frame, &temp_contour_group, 0, {0, 0, 255, 0}, 2)

        // Calculate fitted height and width from the box points
        v.fitted_height = _calculate_distance(box_points[0], box_points[1])
        v.fitted_width = _calculate_distance(box_points[1], box_points[2])

        // Correct for rotation by ensuring fitted_height/width matches known_height/width aspect
        height_is_larger := v.known_height > v.known_width
        width_is_larger := v.fitted_width > v.fitted_height
        
        if (height_is_larger && width_is_larger) || (!height_is_larger && !width_is_larger) {
             v.fitted_height, v.fitted_width = v.fitted_width, v.fitted_height
        }
    } else {
        v.fitted_height = 0
        v.fitted_width = 0
    }
    return
}


demo_distance_tracker :: proc() {
    // --- 1. Setup Video Capture ---
    capture := ocv.new_videocapture()
    defer ocv.videocapture_release(capture)
    
    ocv.videocapture_open(capture, 0, .CAP_ANY)
    if !ocv.videocapture_isopened(capture) {
        fmt.eprintln("Error: Could not open camera.")
        os.exit(1)
    }

    frame := ocv.new_mat()
    defer ocv.mat_release(frame)
    
    // --- 2. Setup Windows and Trackbars ---
    window_name_cam :: "Camera"
    window_name_mask :: "Mask"
    ocv.named_window(window_name_cam)
    ocv.named_window(window_name_mask)
    defer ocv.destroy_window(window_name_cam)
    defer ocv.destroy_window(window_name_mask)

    // HSV range values, controlled by trackbars
    h_low, s_low, v_low := c.int(35), c.int(50), c.int(50)
    h_high, s_high, v_high := c.int(85), c.int(255), c.int(255)

    ocv.create_trackbar("H_low", window_name_mask, &h_low, 179)
    ocv.create_trackbar("S_low", window_name_mask, &s_low, 255)
    ocv.create_trackbar("V_low", window_name_mask, &v_low, 255)
    ocv.create_trackbar("H_high", window_name_mask, &h_high, 179)
    ocv.create_trackbar("S_high", window_name_mask, &s_high, 255)
    ocv.create_trackbar("V_high", window_name_mask, &v_high, 255)

    // --- 3. Initialize Vision object ---
    // These values are placeholders. You should measure a real object.
    // E.g., An object 24 inches away has a height of 120 pixels on screen.
    // The real object is 6 inches wide and 12 inches tall.
    vision_obj := vision_new(
        pixel_height = 120, 
        known_distance = 24, 
        known_width = 6, 
        known_height = 12,
    )
    defer vision_destroy(vision_obj)

    fmt.println(">>> Starting Vision Tracker Demo. Press ESC to exit.")
    
    // --- 4. Main Loop ---
    for {
        if !ocv.videocapture_read(capture, frame) || ocv.mat_isempty(frame) {
            fmt.eprintln("Error: Could not read frame.")
            break
        }
        // Get current HSV values from trackbars
        lowerb := ocv.Scalar{f64(h_low), f64(s_low), f64(v_low), 0}
        upperb := ocv.Scalar{f64(h_high), f64(s_high), f64(v_high), 0}
        
        // Process the frame
        mask := vision_update_frame(vision_obj, frame, lowerb, upperb)
        
        // Get and display results
        distance := vision_get_distance(vision_obj)
        angle := vision_get_angle(vision_obj)

        label := fmt.ctprintf("Dist: %.2f in, Angle: %.2f deg", distance, angle)
        ocv.put_text(frame, label, 10, 30, .SIMPLEX, 1.0, 255, 255, 255, 2)
        
        ocv.image_show(window_name_cam, frame)
        ocv.image_show(window_name_mask, mask)
        
        ocv.mat_release(mask) // Ensure mask is released each loop
        if ocv.wait_key(1) == 27 { // ESC key
            break
        }
    }
}