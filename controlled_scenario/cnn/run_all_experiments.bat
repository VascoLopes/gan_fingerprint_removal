python create_real_to_fake_dataset.py F:\FACE_DATASETS\VGG_FACE_2\byid_alignedlib_0.3_train\ F:\FACE_DATASETS\NVIDIA_FakeFace\byimg_alignedlib_0.3\ F:\gan_fingerprint_removal\data\real2fake_VF2_TPDNE
python ../../train_pytorch_gpu.py F:\gan_fingerprint_removal\data\real2fake_VF2_TPDNE
python ../../test_pytorch_gpu.py F:\gan_fingerprint_removal\data\real2fake_VF2_TPDNE F:\gan_fingerprint_removal\data\real2fake_VF2_TPDNE
python create_real_to_fake_dataset.py F:\FACE_DATASETS\VGG_FACE_2\byid_alignedlib_0.3_train\ F:\FACE_DATASETS\100K_FAKE\byimg_alignedlib_0.3\ F:\gan_fingerprint_removal\data\real2fake_VF2_100F
python ../../train_pytorch_gpu.py F:\gan_fingerprint_removal\data\real2fake_VF2_100F
python ../../test_pytorch_gpu.py F:\gan_fingerprint_removal\data\real2fake_VF2_100F F:\gan_fingerprint_removal\data\real2fake_VF2_100F
python create_real_to_fake_dataset.py F:\FACE_DATASETS\CASIA-WebFace\byid_alignedlib_0.3\ F:\FACE_DATASETS\NVIDIA_FakeFace\byimg_alignedlib_0.3\ F:\gan_fingerprint_removal\data\real2fake_CASIA_TPDNE
python ../../train_pytorch_gpu.py F:\gan_fingerprint_removal\data\real2fake_CASIA_TPDNE
python ../../test_pytorch_gpu.py F:\gan_fingerprint_removal\data\real2fake_CASIA_TPDNE F:\gan_fingerprint_removal\data\real2fake_CASIA_TPDNE
python create_real_to_fake_dataset.py F:\FACE_DATASETS\CASIA-WebFace\byid_alignedlib_0.3\ F:\FACE_DATASETS\100K_FAKE\byimg_alignedlib_0.3\ F:\gan_fingerprint_removal\data\real2fake_CASIA_100F
python ../../train_pytorch_gpu.py F:\gan_fingerprint_removal\data\real2fake_CASIA_100F
python ../../test_pytorch_gpu.py F:\gan_fingerprint_removal\data\real2fake_CASIA_100F F:\gan_fingerprint_removal\data\real2fake_CASIA_100F
