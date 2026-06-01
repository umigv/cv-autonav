from hsv import hsv

hsv_obj = hsv("LEFT_ZED")
hsv_obj.tune("white", use_zed=True)

# hsv_obj = hsv("RIGHT_ZED")
# hsv_obj.tune("white", use_zed=True)