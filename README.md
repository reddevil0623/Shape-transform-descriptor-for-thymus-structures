This repo contains the code for computing ECT, SECT, DETECT, and SampEuler for simplicial complexes and example uses of them.

My_ECT.py contains ECT-based algorithms for simplicial complexes.

Selected_quadrants.zip contains  the TIFF images of the quadrants we used for the classification study in the paper.

Whole_tissue_masks.zip contains a total mask classifying the thymus into cortex regions and medulla regions for all thymi we used in this study. It also has individual binary masks for the thymic epithelial cells (TEC) of each thymus.

all_cortex_quadrants.zip contains all separate cortex quadrant images generated from tiling and filtering through each binary TEC mask.

patch_TEC1hop.csv contains the cell counts for each selected cell type within 5μm to TECs of each cortex quadrant.

shap_example.py contains code as an example showing how we performed SHAP analysis using the vectorized SampEuler as features.

synthetic_network_and_MPEG.ipynb contains the code for how we generated the synthetic network examples, and codes for computing ECT, DETECT, SampEuler, Vectorized SampEuler for these examples, as well as visualizing results using MDScale.

ECT_video.mp4 is a video showing how ECT and SECT works for 2D data.
