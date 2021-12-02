# Edge-Detection-Computer-Vision

* Fistly I have implemented the median filter for denoising the image.
  ![denoise image](edge_detection_denoise.jpg)
* Next implemented convolution operation by flipping the kernel.
* Next I have used sobel_x and sobel_y filters to detect vertical and horizontal edges in the image by using the convolution operation.
  ![Edge along X axis, Vertical Edges](edge_detection_edge_x.jpg)
  ![Edge along Y axis, Horizontal Edges](edge_detection_edge_y.jpg) 
* Computed the magnitude of vertical and horizontal edges and these are the combined edges:
  ![Edges Magnitude](edge_detection_edge_mag.jpg) 
*  Next I have used sobel_45 filter and sobel_135 filter to detect the diagonal edges.
  ![Diagonal Edges](edge_detection_edge_diag1.jpg) 
  ![Diagonal Edges](edge_detection_edge_diag2.jpg) 
