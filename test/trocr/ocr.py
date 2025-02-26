import argparse
import requests
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def load_model():
    """
    Load the TrOCR processor and model, move the model to the available device,
    and return them.
    """
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
    model.eval()  # set the model to evaluation mode

    # Choose device: use MPS on Apple Silicon, CUDA if available, else CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    return processor, model, device


processor, model, device = load_model()


def predict(ImageUrl=None, imgDraw=None, imgUplod=None):
    """
    Process an image from a URL, drawn canvas, or uploaded image,
    and return the recognized text.
    """
    if ImageUrl:
        response = requests.get(ImageUrl, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
    elif imgDraw:
        image = imgDraw.convert("RGB")
    elif imgUplod:
        image = imgUplod.convert("RGB")
    else:
        raise ValueError("No image provided.")

    # Process image to tensor and move it to the appropriate device.
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    # Increase max_length in case the text is long; adjust num_beams for better search.
    generated_ids = model.generate(pixel_values, max_length=512, num_beams=4)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def segment_lines3(image):
    """
    Segments a multi-line document image into individual line images.
    Returns a list of PIL.Image.Image objects, each containing one text line.

    This method:
      1. Converts the image to grayscale.
      2. Uses adaptive thresholding (inverted) to handle uneven illumination.
      3. Applies horizontal dilation to merge text into line groups.
      4. Extracts and sorts contours to crop out each line.

    Returns:
      list of PIL.Image.Image: Cropped images of individual text lines.
    """
    import cv2
    import numpy as np
    from PIL import Image

    # Convert the PIL image to grayscale (OpenCV format)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Apply adaptive thresholding (invert so text becomes white on black)
    # Adjust blockSize and C based on your image characteristics
    bin_inv = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,  # smaller blockSize can capture finer details
        C=10
    )

    # Use a horizontal kernel to connect text components in the same line.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3))
    dilated = cv2.dilate(bin_inv, kernel, iterations=1)

    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out contours that are too small (noise) or too narrow
        if h < 15 or w < 50:
            continue
        boxes.append((x, y, w, h))

    # Sort the bounding boxes from top to bottom
    boxes.sort(key=lambda b: b[1])

    # Convert the original image to OpenCV BGR format for cropping
    color_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    line_images = []
    for (x, y, w, h) in boxes:
        crop = color_cv[y:y + h, x:x + w]
        line_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        line_images.append(line_img)

    # If no lines are detected, return the whole image as one line
    if not line_images:
        return [image]

    return line_images


def segment_lines2(image):
    """
    Segments a (potentially) multi-line handwritten or scanned document image into individual line images.
    Returns a list of PIL.Image.Image objects, each containing one line.

    This method:
      1. Converts the image to grayscale.
      2. Performs adaptive thresholding (inverted).
      3. Applies a horizontal morphological dilation to group words into lines.
      4. Finds contours and crops each line from the original (color) image.

    Requirements:
      - pip install opencv-python-headless (or opencv-python)
      - pip install pillow
      - import cv2, numpy, PIL

    Args:
      image (PIL.Image.Image): Input image (RGB or grayscale).

    Returns:
      list of PIL.Image.Image: Each item is a cropped image of a single text line.
    """
    import cv2
    import numpy as np
    from PIL import Image

    # Convert PIL to OpenCV grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Adaptive threshold (invert so text is white on black)
    bin_inv = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,  # adjust as needed
        C=15
    )

    # Morphological dilation to connect words into full lines
    # Adjust kernel size as needed for your text size/spacing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2))
    dilated = cv2.dilate(bin_inv, kernel, iterations=1)

    # Find contours (each contour ideally corresponds to a line)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the original PIL image to OpenCV color (so crops preserve color)
    color_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Build a list of bounding boxes, ignoring very small boxes
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 10:  # skip tiny boxes
            continue
        boxes.append((x, y, w, h))

    # Sort boxes top-to-bottom by their y-coordinate
    boxes.sort(key=lambda b: b[1])

    # Crop each line from the color image
    line_images = []
    for (x, y, w, h) in boxes:
        crop = color_cv[y:y+h, x:x+w]
        # Convert back to PIL for downstream processing
        line_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        line_images.append(line_img)

    # If nothing found, return the entire image as a single line
    if not line_images:
        return [image]

    return line_images


def segment_lines(
    image: Image.Image,
    min_line_height: int = 10,
    gap_threshold: int = 10,
    morph_kernel_size: int = 3
):
    """
    Splits the input image into separate line images using a horizontal projection.
    Uses Otsu's threshold + morphological operations to make line segmentation more robust.

    Args:
      image (PIL.Image.Image): Input image.
      min_line_height (int): Minimum height of a text line to be considered valid.
      gap_threshold (int): Row-sum threshold under which we consider a row "empty."
      morph_kernel_size (int): Kernel size for morphological dilation to connect text.

    Returns:
      list of PIL.Image.Image: A list of cropped images, each for one text line.
    """
    # Convert PIL image to a grayscale OpenCV image
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Use Otsu's threshold to handle different lighting conditions
    # text will be white (255) on black background after THRESH_BINARY_INV
    _, binary = cv2.threshold(image_cv, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Optional: Morphological dilation to connect text areas better
    # This helps ensure each line is more continuous and separated from adjacent lines
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Sum pixel values along each row to create a horizontal projection
    row_sum = np.sum(binary, axis=1)

    # Scan row by row, detecting transitions from blank -> text -> blank
    lines = []
    in_line = False
    start_row = 0
    for i, value in enumerate(row_sum):
        # "value" is how many white pixels are in row i
        # If it's below gap_threshold, consider that row effectively blank
        if not in_line and value >= gap_threshold:
            in_line = True
            start_row = i
        elif in_line and value < gap_threshold:
            in_line = False
            # If the line is tall enough, store it
            if i - start_row > min_line_height:
                lines.append((start_row, i))

    # In case the text goes until the bottom
    if in_line:
        lines.append((start_row, len(row_sum)))

    # Crop the original image to each detected line
    line_images = []
    for (start, end) in lines:
        cropped = image.crop((0, start, image.width, end))
        line_images.append(cropped)

    return line_images

def deskew_and_segment_lines_simple(image: Image.Image, deskew_dilate_iter=1, min_line_height=10):
    """
    A more robust line segmentation method:
      1. Convert to grayscale + binarize (Otsu).
      2. Deskew the image so lines are horizontal.
      3. Morphologically dilate horizontally to merge words into full text lines.
      4. Find contours, each contour = bounding box of a line.
      5. Crop each bounding box and return as separate PIL images.

    Args:
      image (PIL.Image.Image): The input image.
      deskew_dilate_iter (int): Number of dilation iterations used when deskewing.
      min_line_height (int): Minimum height for a line bounding box to be considered valid.

    Returns:
      list of PIL.Image.Image: Cropped images, each representing one text line.
    """

    # Convert to OpenCV grayscale
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Binarize (white text on black background) using Otsu
    # We invert because it’s easier to find contours of white text on black
    _, bin_inv = cv2.threshold(image_cv, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 1) DESKEW
    # Find all foreground (white) coordinates
    coords = np.column_stack(np.where(bin_inv > 0))
    # If the document is entirely blank, return the original as is
    if len(coords) == 0:
        return [image]

    # Calculate the angle of the minimum area bounding rectangle
    angle = cv2.minAreaRect(coords)[-1]
    # Adjust angle to the correct deskew direction
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Rotate the image to deskew
    (h, w) = bin_inv.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    deskewed = cv2.warpAffine(bin_inv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Optionally dilate the deskewed image to ensure text lines are more continuous
    # This helps with line-level bounding boxes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    deskewed = cv2.dilate(deskewed, kernel, iterations=deskew_dilate_iter)

    # 2) FIND LINE BOUNDING BOXES
    # Each contour should (ideally) represent one line of text
    contours, _ = cv2.findContours(deskewed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the PIL image again to deskew the original color image in sync
    # so that the line crops match the deskewed coordinates
    color_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    color_cv = cv2.warpAffine(color_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    line_images = []
    boxes = []

    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        # Filter out very small lines
        if h_box < min_line_height:
            continue
        boxes.append((x, y, w_box, h_box))

    # Sort bounding boxes top to bottom
    boxes = sorted(boxes, key=lambda b: b[1])

    for (x, y, w_box, h_box) in boxes:
        # Crop the deskewed color image
        crop = color_cv[y:y+h_box, x:x+w_box]
        # Convert back to PIL for further processing
        line_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        line_images.append(line_img)

    # If no lines found, return original
    if not line_images:
        return [image]

    return line_images
def deskew_and_segment_lines(
    image: Image.Image,
    scale_factor: float = 1.0,
    min_line_height: int = 15,
    morph_kernel_width: int = 30,
    morph_kernel_height: int = 2,
    deskew_dilate_iter: int = 1
):
    """
    A more robust line segmentation method:
      1. (Optional) Scale the image up/down if text is small/large.
      2. Convert to grayscale + adaptive threshold (inverted).
      3. Deskew the image so lines are horizontal.
      4. Morphologically dilate horizontally to merge words into full text lines.
      5. Find contours, each contour ~ bounding box of a line.
      6. Crop each bounding box from the *deskewed* color image.

    Args:
      image (PIL.Image.Image): The input image.
      scale_factor (float): Scale the image before processing if text is very small or large.
      min_line_height (int): Minimum height for a line bounding box to be considered valid.
      morph_kernel_width (int): Horizontal size for morphological dilation kernel.
      morph_kernel_height (int): Vertical size for morphological dilation kernel.
      deskew_dilate_iter (int): Number of dilation iterations used when deskewing.

    Returns:
      list of PIL.Image.Image: Cropped images, each representing one text line.
    """
    # Optionally scale the image to help with small text
    if scale_factor != 1.0:
        new_w = int(image.width * scale_factor)
        new_h = int(image.height * scale_factor)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Convert to OpenCV grayscale
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # ADAPTIVE THRESHOLD to handle uneven illumination on older scans
    # We invert so text becomes white (255) on black (0).
    # "blockSize" and "C" can be tuned for your scans.
    bin_inv = cv2.adaptiveThreshold(
        image_cv, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=31,  # bigger blockSize => more local smoothing
        C=15
    )

    # 1) DESKEW
    coords = np.column_stack(np.where(bin_inv > 0))
    if len(coords) == 0:
        # If the document is entirely blank or thresholded incorrectly, return the original
        return [image]

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = bin_inv.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    deskewed = cv2.warpAffine(bin_inv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 2) MORPHOLOGICAL DILATION
    # This merges letters/words on the same line into a single contour
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel_width, morph_kernel_height)
    )
    deskewed = cv2.dilate(deskewed, kernel, iterations=deskew_dilate_iter)

    # 3) FIND LINE BOUNDING BOXES
    contours, _ = cv2.findContours(deskewed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Deskew the original (scaled) color image in sync
    color_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    color_cv = cv2.warpAffine(color_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    boxes = []
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        if h_box < min_line_height:
            continue
        boxes.append((x, y, w_box, h_box))

    # Sort bounding boxes top to bottom
    boxes = sorted(boxes, key=lambda b: b[1])

    line_images = []
    for (x, y, w_box, h_box) in boxes:
        crop = color_cv[y:y+h_box, x:x+w_box]
        line_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        line_images.append(line_img)

    if not line_images:
        # fallback if no lines found
        return [image]

    return line_images
def main():
    parser = argparse.ArgumentParser(
        description="Recognize handwritten text from an image using TrOCR."
    )
    parser.add_argument("--image_path", type=str, default=None, help="Local image file path.")
    parser.add_argument("--url", type=str, default=None, help="URL of the image.")
    args = parser.parse_args()

    if not args.image_path and not args.url:
        raise ValueError("No image input provided. Please specify either --image_path or --url.")

    if args.url:
        response = requests.get(args.url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
    else:
        image = Image.open(args.image_path).convert("RGB")

    # Segment lines. Adjust parameters as needed:
    # - scale_factor: If text is tiny, try 1.5 or 2.0. If huge, try < 1.0.
    # - morph_kernel_width: If lines merge, decrease. If lines split, increase.
    # line_images = deskew_and_segment_lines(
    #     image,
    #     scale_factor=1.0,         # Try 1.5 or 2.0 if text is small
    #     min_line_height=10,
    #     morph_kernel_width=30,    # Increase if lines are broken up
    #     morph_kernel_height=2,
    #     deskew_dilate_iter=1
    # )

    line_images = deskew_and_segment_lines_simple(image)

    # line_images = segment_lines3(image)

    print("Number of segmented lines:", len(line_images))
    # If segmentation fails (returns an empty list), assume the whole image is one line.
    if not line_images:
        line_images = [image]

    recognized_lines = []
    for idx, line_img in enumerate(line_images):
        text = predict(imgUplod=line_img)
        recognized_lines.append(text)

    # Join the recognized text for each line with newline characters.
    final_text = "\n".join(recognized_lines)
    print("Recognized text:\n", final_text)


if __name__ == '__main__':
    main()

# Usage
# Using a URL:
# python ocr.py --url "https://example.com/path/to/image.jpg"

# Using a Local Image Path:
# python ocr.py --image_path "/path/to/local/image.jpg"
