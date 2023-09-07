def create_binary_mask(image_in):
    height, width = image_in.shape
    white_image = np.ones((height, width), dtype=np.uint8) * 255

    top_black_height = int(0.50 * height)
    left_black_width = int(0.25 * width)
    right_black_width = int(0.25 * width)

    white_image[0:top_black_height, :] = 0
    white_image[:, 0:left_black_width] = 0
    white_image[:, -right_black_width:] = 0

    return white_image
