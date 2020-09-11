from utils import create_input_files

if __name__=='__main__':
  # Create input files (along with word map)
  create_input_files(dataset='3dcc',
                     data_json_path='../3d_change_captioning/dataset/3dcc_v0-4/captions/',
                     img_feature_path='../VQA/CLEVR/clevr-iep-master/data/3DCC_v0-4_12view/splited_files/',
                     captions_per_image = 5,
                     min_word_freq = 0,
                     output_folder = '../3d_change_captioning/dataset/for_train/3dcc_v0-4/duda_v3/',
                     viewpoint=3,
                     max_len=20)
