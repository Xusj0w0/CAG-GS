python cag_gs/tools/convert_dataset.py \
    datasets/from_citygs/building \
    datasets/MegaNeRF/building \
    --down_sample_factor 4

python cag_gs/tools/convert_dataset.py \
    datasets/from_citygs/rubble \
    datasets/MegaNeRF/rubble \
    --down_sample_factor 4

python cag_gs/tools/convert_dataset.py \
    datasets/from_citygs/residence \
    datasets/MegaNeRF/residence \
    --down_sample_factor 4

python cag_gs/tools/convert_dataset.py \
    datasets/from_citygs/sciart \
    datasets/MegaNeRF/sci-art \
    --down_sample_factor 4

python cag_gs/tools/convert_dataset.py \
    datasets/from_citygs/matrix_city_aerial \
    datasets/MatrixCity/smallcity \
    --prefix "val_" \
    --rescale_width 1600