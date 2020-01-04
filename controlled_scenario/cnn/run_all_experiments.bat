python create_real_to_fake_dataset.py E:\FACE_DATASETS\VGG_FACE_2\byid_alignedlib_0.3_train\ E:\FACE_DATASETS\NVIDIA_FakeFace\byimg_alignedlib_0.3\ E:\gan_fingerprint_removal\data\real2fake_VF2_TPDNE
python ../../train_pytorch_gpu.py E:\gan_fingerprint_removal\data\real2fake_VF2_TPDNE
python ../../test_pytorch_gpu.py E:\gan_fingerprint_removal\data\real2fake_VF2_TPDNE E:\gan_fingerprint_removal\data\real2fake_VF2_TPDNE

