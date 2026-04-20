# Preprocessing (Lura)

### ROI
the paper going over the challenge mentions all 8 teams used ROI. DatasetCompressor.ipynb should be modified so that a better region of interest is captured. The current method used just has the same bounding box for every image but a method to crop it so all regions with black pixels are cropped out. the regions depend image by image so a function should be implemented to find the correct bounding box per image. Looking for methods on how to do this online would be a good start. End result should still be resized to 1028x1028

### Looking at classes
The CSV ground truths file has a column for every disease, a way to see the disease counts could be to just sum for every column to get a good class count. Figuring out how to do the weights might be tricky as it may need to be on a per disease basis?



# 