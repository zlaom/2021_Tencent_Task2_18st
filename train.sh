mkdir /home/tione/notebook/dataset

# generate features for nextvlad
python pre/cait_train_feature.py
python pre/cait_val_feature.py
python pre/cait_test_feature.py


# -----------------preprocess-----------------------------------
# generate frames
python pre/train_frames.py
python pre/val_frames.py


# generate input transform tensors
python pre/train_transform.py
python pre/val_transform.py


# ---------------------train-------------------------------------
# train x3d and nextvlad
python src/x3d.py
python src/nextvlad.py

# get features(last hidden layer) of x3d
python src/x3d_train_feature.py
python src/x3d_val_feature.py

# get features(last hidden layer) of nextvlad
python src/nextvlad_train_feature.py
python src/nextvlad_val_feature.py

# train fusion nextvlad+x3d and text+x3d
python src/fusion_nv.py
python src/fusion_tv.py


# --------------------test---------------
# preprocess
python pre/test_frames.py
python pre/test_transform.py


# get_feature
python src/x3d_test_feature.py
python src/nextvlad_test_feature.py


# get fusion val.json files
python src/fusion_nv_val.py
python src/fusion_tv_val.py

# get fusion test.json files
python src/fusion_nv_test.py
python src/fusion_tv_test.py


