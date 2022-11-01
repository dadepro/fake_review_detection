## Code and data for "Detecting Fake Review Buyers Using Network Structure: Direct Evidence from Amazon"

### Datasets description

There are five datasets that are needed for replication:

1. **product_level_data_text_feats.csv.gz**: Product-level data that contains top-1000 TF-IDF features along with product ID and fake review product label (fake = 1 if the product buys fake reviews, 0 otherwise.) 
2. **product_level_data_with_img_feats.csv.gz**: Product-level data that contains metadata, network, and image features along with product ID and fake review product label (fake = 1 if the product buys fake reviews, 0 otherwise.)
3. **product_level_data_without_img_feats.csv.gz**: Product-level data that contains metadata and network features along with product ID and fake review product label (fake = 1 if the product buys fake reviews, 0 otherwise.)
4. **UCSD_product_level_data.csv.gz**: Product-level data that contain metadata and network features of products in the home and kitchen category part of the UCSD [Amazon Review Data](https://nijianmo.github.io/amazon/index.html).
5. **UCSD_home_and_kitchen_reviews.csv.gz** (due to its large size (~ 1GB), this dataset needs to be downloaded using this [link](https://www.dropbox.com/s/o2jv9uw7emd0dgy/UCSD_home_and_kitchen_reviews.csv.gz?dl=0)):  Review-level data that contains home and kitchen reviews of the products used to obtain product-level data in dataset 4 above. 

See Table 1 in the [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4147920) for a description of the features.

### Code
The code is written with Python 3.9.12. 

Under the folder "code", there are two scripts:

1. **classification.py** which will reproduce the supervised results discussed in the paper. It equires the product-level data (datasets 1 and 2) that contain the metadata, image, network, and text features included in the "data" folder. To run the script simply type:
    
  ```
   python classification.py
  ```
2.  **clustering.py** which will reproduce the unsupervised results discussed in the paper. It requires the product-level data (datasets 3 and 4) and review-level data (dataset 5) that can be downloaded from [here](https://www.dropbox.com/s/o2jv9uw7emd0dgy/UCSD_home_and_kitchen_reviews.csv.gz?dl=0). Tu run the script, simply type:
  ```
   python clustering.py
  ```

**Reference**: He, S., Hollenbeck, B., Overgoor, G., Proserpio, D., & Tosyali, A. (2022). Detecting Fake Review Buyers Using Network Structure: Direct Evidence from Amazon. The paper can be downloaded [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4147920).
