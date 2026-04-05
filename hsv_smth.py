import pyzed.sl as sl
import cv2
import numpy as np

# Dummy callback function required by OpenCV trackbars
def empty_callback(value):
    pass

def main():
    # 1. Initialize the ZED Camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720 # Adjust resolution as needed
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NONE # Disable depth to save compute for tuning

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {err}")
        return

    # 2. Set the requested camera parameters
    print("Applying custom camera parameters...")
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, 1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 6)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 8)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.HUE, 7)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 4)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 4)

    # 3. Setup the OpenCV Control Window
    controls_window = "HSV Controls"
    cv2.namedWindow(controls_window)
    cv2.resizeWindow(controls_window, 400, 250)

    # Note: OpenCV Hue range is 0-179. Saturation and Value are 0-255.
    cv2.createTrackbar("H Min", controls_window, 0, 179, empty_callback)
    cv2.createTrackbar("S Min", controls_window, 0, 255, empty_callback)
    cv2.createTrackbar("V Min", controls_window, 0, 255, empty_callback)
    cv2.createTrackbar("H Max", controls_window, 179, 179, empty_callback)
    cv2.createTrackbar("S Max", controls_window, 255, 255, empty_callback)
    cv2.createTrackbar("V Max", controls_window, 255, 255, empty_callback)

    image_zed = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    print("Starting video stream. Press 'q' to quit and print final values.")

    while True:
        # Grab a new frame
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left camera image
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            
            # Convert ZED image format to OpenCV format
            # ZED returns BGRA, we need BGR for standard OpenCV processing
            image_bgra = image_zed.get_data()
            image_bgr = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR)
            
            # Convert the BGR image to HSV
            image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

            # Read the current positions of all trackbars
            h_min = cv2.getTrackbarPos("H Min", controls_window)
            s_min = cv2.getTrackbarPos("S Min", controls_window)
            v_min = cv2.getTrackbarPos("V Min", controls_window)
            h_max = cv2.getTrackbarPos("H Max", controls_window)
            s_max = cv2.getTrackbarPos("S Max", controls_window)
            v_max = cv2.getTrackbarPos("V Max", controls_window)

            # Define the upper and lower bounds based on slider inputs
            lower_bound = np.array([h_min, s_min, v_min])
            upper_bound = np.array([h_max, s_max, v_max])

            # Generate the black and white mask
            mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

            # Display the results side-by-side (or in separate windows)
            cv2.imshow("Raw Image (Left Camera)", image_bgr)
            cv2.imshow("HSV Mask", mask)

            # Break the loop if 'q' is pressed
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("\n--- Final HSV Values ---")
                print(f"Lower Bound: [{h_min}, {s_min}, {v_min}]")
                print(f"Upper Bound: [{h_max}, {s_max}, {v_max}]")
                break

    # 4. Clean up resources
    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()