package main

import "core:c"
import "core:fmt"
import "core:os"
import "core:strings"
import "../ocv"

/*
    load_class_names: A robust function to load class names from a file.

    This implementation correctly handles memory by creating an owned `clone` of each
    line. The caller is now responsible for freeing the memory for each string's data
    and for the array that holds them. This prevents use-after-free errors.
 */
load_class_names :: proc(filename: string) -> (names: []string, ok: bool) {
    // 1. Read the entire file into a buffer.
    contents_bytes, read_ok := os.read_entire_file_from_filename(filename, context.allocator)
    if !read_ok {
        fmt.eprintfln("Error: Could not read class names file '%s'", filename)
        return nil, false
    }

    // 2. Create a temporary dynamic array to store the owned strings.
    //    This is easier than counting valid lines in a first pass.
    owned_names_dyn := make([dynamic]string)

    // 3. Iterate over the buffer, creating owned clones of each valid line.
    //    Using a split_iterator is efficient as it doesn't create an intermediate slice of views.
    tmp := string(contents_bytes)
    for line in strings.split_iterator(&tmp, "\n") {
        trimmed_line := strings.trim_space(line)
        if len(trimmed_line) > 0 {
            // Clone the line and append the new, owned string to our dynamic array.
            append(&owned_names_dyn, strings.clone(trimmed_line))
        }
    }
    // delete(tmp) // Clean up the temporary string to avoid dangling references.
    // 4. CRITICAL: Now that we are completely done with views into the buffer,
    //    we can safely delete it.
    delete(contents_bytes)

    // 5. Convert the dynamic array to a simple, owned slice to return to the caller.
    //    The caller will be responsible for deleting this slice and the strings within it.
    final_slice := owned_names_dyn[:]

    // 6. Clean up the temporary dynamic array structure.
    delete(owned_names_dyn)

    return final_slice, true
}

demo_face_detection :: proc() {
    // --- 1. Initialize Video, Classifier, and Mats ---
    capture := ocv.new_videocapture()
    if capture == nil {
        fmt.eprintln("Error: could not create a video capture.")
        os.exit(1)
    }
    // `ocv.videocapture_release` is the correct and only deallocation needed.
    defer ocv.videocapture_release(capture)
    
    device_id: c.int = 0
    ocv.videocapture_open(capture, device_id, .CAP_ANY)
    if !ocv.videocapture_isopened(capture) {
        fmt.eprintfln("Error: can't open camera stream.")
        os.exit(1)
    }

    cascade_path :: "assets/haarcascade_frontalface.xml"
    classifier := ocv.cascade_classifier_new()
    if classifier == nil {
        fmt.eprintln("Error: could not create a cascade classifier.")
        os.exit(1)
    }
    defer ocv.cascade_classifier_release(classifier)
    if !ocv.cascade_classifier_load(classifier, cascade_path) {
        fmt.eprintfln("Error: could not load classifier from file: %s", cascade_path)
        os.exit(1)
    }

    frame := ocv.new_mat()
    defer ocv.mat_release(frame)
    gray_frame := ocv.new_mat()
    defer ocv.mat_release(gray_frame)

    window_name :: "Face Detection Demo"
    ocv.named_window(window_name)
    defer ocv.destroy_window(window_name)

    // --- 2. Initialize Kalman Filter and Tracking State ---
    // Note: The original code re-assigned kf to a new filter, leaking the first one.
    kf := ocv.kalmanfilter_new()
    defer ocv.kalmanfilter_release(kf)

    is_tracking := false
    frames_since_detection := 0
    MAX_MISSED_FRAMES :: 10

    fmt.println(">>> Starting face detection... Press ESC to exit.")
    
    // --- 3. Main Loop: Read, Predict, Detect, Correct, Draw ---
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
        
        predicted_rect := ocv.kalmanfilter_predict(kf)
        detected_faces_ptr := ocv.cascade_classifier_detect_multi_scale(classifier, gray_frame)
        
        face_found := false
        if detected_faces_ptr != nil {
            defer ocv.free_rects(detected_faces_ptr)
            if detected_faces_ptr.count > 0 {
                face_found = true
                rect_slice := ([^]ocv.Rect)(detected_faces_ptr.rects)[:detected_faces_ptr.count]
                measurement_rect := rect_slice[0]

                if !is_tracking {
                    fmt.println("First detection! Initializing Kalman Filter state.")
                    ocv.kalmanfilter_set_state_post(kf, measurement_rect)
                    is_tracking = true
                }
                
                corrected_rect := ocv.kalmanfilter_correct(kf, measurement_rect)
                ocv.rectangle(frame, corrected_rect, 0, 255, 0, 2)
                ocv.put_text(frame, "Face (Corrected)", corrected_rect.x, corrected_rect.y - 10, 
                            .SIMPLEX, 0.7, 0, 255, 0, 2)
                frames_since_detection = 0
            }
        }
        
        if !face_found {
            frames_since_detection += 1
            if is_tracking && frames_since_detection < MAX_MISSED_FRAMES {
                ocv.rectangle(frame, predicted_rect, 255, 100, 0, 2)
                ocv.put_text(frame, "Face (Predicted)", predicted_rect.x, predicted_rect.y - 10,
                            .SIMPLEX, 0.7, 255, 100, 0, 2)
            } else {
                if is_tracking {
                    fmt.println("Lost track.")
                }
                is_tracking = false
            }
        }
        
        ocv.image_show(window_name, frame)

        if ocv.wait_key(25) == 27 { // ESC key
            break
        }
    }
}