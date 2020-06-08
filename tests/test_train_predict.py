"""Testing saving and loading model"""
# pylint: disable=invalid-name

# pylint: disable=wrong-import-position
import os
import sys
import time
import shutil
import glob
import unittest

#from src.train import get_parser
sys.path.append("/BrainSeg/src/")
import train
import predict
# pylint: enable=wrong-import-position

def rmdir(dir_path: str) -> None:
    """Sleep-loop for removing a directory due to stale NFS files"""
    shutil.rmtree(dir_path, ignore_errors=True)
    while os.path.exists(dir_path):
        time.sleep(.1)
        shutil.rmtree(dir_path, ignore_errors=True)

class TestUNetFullModelLoading(unittest.TestCase):
    """Testing UNet full model loading"""
    def setUp(self):
        """Setup shared by all tests"""
        self.train_parser = train.get_parser()
        self.predict_parser = predict.get_parser()
        self.ckpt_dir = '/BrainSeg/tmp_checkpoints'
        self.log_dir = '/BrainSeg/tmp_tf_logs'
        self.save_dir = '/BrainSeg/tmp_outputs'

    def test_UNet_Zero_Pad_1024_SCCE(self):
        """Tests UNet_Zero_Pad 1024 SCCE"""
        train_args = self.train_parser.parse_args(
            ['UNet_Zero_Pad', '--patch-size=1024', '--loss-func=SCCE',
             '--batch-size=2', '--num-epochs=1',
             '--steps-per-epoch=5', '--val-steps=5',
             f'--ckpt-dir={self.ckpt_dir}', f'--log-dir={self.log_dir}',
             '--file-suffix=unittest'])
        train.train(train_args)

        # Find the checkpoint file
        ckpt_file = glob.glob(os.path.join(
            glob.glob(os.path.join(self.ckpt_dir, '*'))[0], '*'))[0]

        predict_args = self.predict_parser.parse_args(
            [ckpt_file, train_args.data_dir_AD, train_args.data_dir_control,
             '--patch-size=1024', '--batch-size=2',
             f'--save-dir={self.save_dir}',
             '--test-svs-idx=0', '--predict-one-round'])
        predict.predict(predict_args)

    def test_UNet_Zero_Pad_1024_BSCCE(self):
        """Tests UNet_Zero_Pad 1024 BSCCE"""
        train_args = self.train_parser.parse_args(
            ['UNet_Zero_Pad', '--patch-size=1024', '--loss-func=BSCCE',
             '--batch-size=2', '--num-epochs=1',
             '--steps-per-epoch=5', '--val-steps=5',
             f'--ckpt-dir={self.ckpt_dir}', f'--log-dir={self.log_dir}',
             '--file-suffix=unittest'])
        train.train(train_args)

        # Find the checkpoint file
        ckpt_file = glob.glob(os.path.join(
            glob.glob(os.path.join(self.ckpt_dir, '*'))[0], '*'))[0]

        predict_args = self.predict_parser.parse_args(
            [ckpt_file, train_args.data_dir_AD, train_args.data_dir_control,
             '--patch-size=1024', '--batch-size=2',
             f'--save-dir={self.save_dir}',
             '--test-svs-idx=0', '--predict-one-round'])
        predict.predict(predict_args)

    def test_UNet_Zero_Pad_1024_Sparse_Focal(self):
        """Tests UNet_Zero_Pad 1024 Sparse_Focal"""
        train_args = self.train_parser.parse_args(
            ['UNet_Zero_Pad', '--patch-size=1024', '--loss-func=Sparse_Focal',
             '--batch-size=2', '--num-epochs=1',
             '--steps-per-epoch=5', '--val-steps=5',
             f'--ckpt-dir={self.ckpt_dir}', f'--log-dir={self.log_dir}',
             '--file-suffix=unittest'])
        train.train(train_args)

        # Find the checkpoint file
        ckpt_file = glob.glob(os.path.join(
            glob.glob(os.path.join(self.ckpt_dir, '*'))[0], '*'))[0]

        predict_args = self.predict_parser.parse_args(
            [ckpt_file, train_args.data_dir_AD, train_args.data_dir_control,
             '--patch-size=1024', '--batch-size=2',
             f'--save-dir={self.save_dir}',
             '--test-svs-idx=0', '--predict-one-round'])
        predict.predict(predict_args)

    def test_UNet_Zero_Pad_1024_Balanced_Sparse_Focal(self):
        """Tests UNet_Zero_Pad 1024 Balanced_Sparse_Focal"""
        train_args = self.train_parser.parse_args(
            ['UNet_Zero_Pad', '--patch-size=1024', '--loss-func=Balanced_Sparse_Focal',
             '--batch-size=2', '--num-epochs=1',
             '--steps-per-epoch=5', '--val-steps=5',
             f'--ckpt-dir={self.ckpt_dir}', f'--log-dir={self.log_dir}',
             '--file-suffix=unittest'])
        train.train(train_args)

        # Find the checkpoint file
        ckpt_file = glob.glob(os.path.join(
            glob.glob(os.path.join(self.ckpt_dir, '*'))[0], '*'))[0]

        predict_args = self.predict_parser.parse_args(
            [ckpt_file, train_args.data_dir_AD, train_args.data_dir_control,
             '--patch-size=1024', '--batch-size=2',
             f'--save-dir={self.save_dir}',
             '--test-svs-idx=0', '--predict-one-round'])
        predict.predict(predict_args)

    def tearDown(self):
        """Teardown shared by all tests"""
        rmdir(self.ckpt_dir)
        rmdir(self.log_dir)
        rmdir(self.save_dir)

if __name__ == '__main__':
    unittest.main()

# pylint: enable=invalid-name
