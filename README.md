## Code and data for the paper entitled *Detecting Fake Review Buyers Using Network Structure: Direct Evidence from Amazon

### Datasets description

### Code
The code is written with Python 3.9.12. 

Under the folder "code", there are two scripts:

1. **classification.py** which will reproduce the supervised results discussed in the paper. It equires the product-level data that contain the metadata, image, network, and text features included in the "data" folder. To run the script simply type:
    
  ```
   python classification.py
  ```
2.  **clustering.py** which will reproduce the unsupervised results discussed in the paper. It requires the product-level data included in the "data" folder and an additional file containing review-level data that can be downloaded from [here](https://www.dropbox.com/s/o2jv9uw7emd0dgy/UCSD_home_and_kitchen_reviews.csv.gz?dl=0). Tu run the script, simply type:
  ```
   python clustering.py
  ```

**Reference**: He, S., Hollenbeck, B., Overgoor, G., Proserpio, D., & Tosyali, A. (2022). Detecting Fake Review Buyers Using Network Structure: Direct Evidence from Amazon. Available at SSRN.
