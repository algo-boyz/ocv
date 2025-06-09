package main

import "core:fmt"
import "core:os"
import "core:slice"
import "../ocv"

demo_object_detection :: proc() {
    // --- 1. Define Constants and Load Resources ---
    MODEL_CONFIG :: "assets/yolov3-tiny.cfg"
    MODEL_WEIGHTS :: "assets/yolov3-tiny.weights"
    CLASS_NAMES_FILE :: "assets/coco.names"
    
    CONF_THRESHOLD :: 0.5
    NMS_THRESHOLD :: 0.4

    class_names, ok := load_class_names(CLASS_NAMES_FILE)
    if !ok { 
        fmt.eprintfln("Error: Could not load class names from %s", CLASS_NAMES_FILE)
        os.exit(1) 
    }
    defer {
        // for name in class_names {
        //     delete(name)
        // }
        // delete(class_names)
    }

    net := ocv.dnn_read_net(MODEL_WEIGHTS, MODEL_CONFIG)
    if net == nil {
        fmt.eprintfln("Error: Could not load network.")
        os.exit(1)
    }
    defer ocv.net_release(net)

    capture := ocv.new_videocapture()
    if capture == nil {
        fmt.eprintln("Error: Could not create VideoCapture.")
        os.exit(1)
    }
    defer ocv.videocapture_release(capture)
    
    ocv.videocapture_open(capture, 0, .CAP_ANY)
    if !ocv.videocapture_isopened(capture) {
        fmt.eprintln("Error: Can't open camera stream.")
        os.exit(1)
    }

    frame := ocv.new_mat()
    if frame == nil {
        fmt.eprintln("Error: Could not create Mat.")
        os.exit(1)
    }
    defer ocv.mat_release(frame)

    window_name :: "Object Detection (YOLOv3-tiny)"
    ocv.named_window(window_name)
    defer ocv.destroy_window(window_name)

    fmt.println(">>> Starting DNN Object Detection... Press ESC to exit.")

    // --- 3. Main Processing Loop ---
    for {
        if !ocv.videocapture_read(capture, frame) || ocv.mat_isempty(frame) {
            fmt.eprintln("Error: Could not read frame.")
            break
        }

        detections := ocv.dnn_detect_objects(net, frame, CONF_THRESHOLD, NMS_THRESHOLD)
        
        if detections != nil {
            // CORRECT: Defer freeing the memory to prevent leaks.
            defer ocv.free_detections(detections)

            fmt.printf("Detected objects: %d\n", detections.count)
            
            for detection in slice.from_ptr(detections.detections, int(detections.count)) {
                fmt.printf("%v", detection)
                rect := detection.box
                class_id := detection.classId
                
                if class_id < 0 { continue }

                ocv.rectangle(frame, rect, 0, 255, 0, 2)

                // label := fmt.ctprintf("%d: %.2f%%", class_id, detection.confidence * 100)
                // ocv.put_text(frame, label, rect.x, rect.y - 10, .SIMPLEX, 0.6, 0, 255, 0, 2)
                // delete(label)
            }
        }
        
        ocv.image_show(window_name, frame)
        if ocv.wait_key(1) == 27 { // ESC key
            break
        }
    }
}
