## Code and data for "Detecting Fake Review Buyers Using Network Structure: Direct Evidence from Amazon"

### Datasets description

There are five datasets that are needed for replication:

1. **product_level_data_text_feats.csv.gz**: 
2. **product_level_data_with_img_feats.csv.gz**: 
3. **product_level_data_without_img_feats.csv.gz**: 
4. **UCSD_product_level_data.csv.gz**: 
5. **UCSD_home_and_kitchen_reviews.csv.gz** (due to its large size (~ 1GB), this dataset needs to be downloaded using this [link](https://www.dropbox.com/s/o2jv9uw7emd0dgy/UCSD_home_and_kitchen_reviews.csv.gz?dl=0)): 

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

**Reference**: He, S., Hollenbeck, B., Overgoor, G., Proserpio, D., & Tosyali, A. (2022). Detecting Fake Review Buyers Using Network Structure: Direct Evidence from Amazon. The paper can be downloaded [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4147920).
