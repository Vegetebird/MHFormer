import numpy as np


class ChunkedGenerator:
    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length=1, pad=0, causal_shift=0,
                 shuffle=False, random_seed=1234,
                 augment=False, reverse_aug= False,kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, out_all = False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)

        pairs = []
        self.saved_index = {}
        start_index = 0

        for key in poses_2d.keys():
            assert poses_3d is None or poses_3d[key].shape[0] == poses_3d[key].shape[0]
            n_chunks = (poses_2d[key].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[key].shape[0]) // 2
            bounds = np.arange(n_chunks + 1) * chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            reverse_augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            keys = np.tile(np.array(key).reshape([1,3]),(len(bounds - 1),1))
            pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector,reverse_augment_vector))
            if reverse_aug:
                pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, ~reverse_augment_vector))
            if augment:
                if reverse_aug:
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector,~reverse_augment_vector))
                else:
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, reverse_augment_vector))

            end_index = start_index + poses_3d[key].shape[0]
            self.saved_index[key] = [start_index,end_index]
            start_index = start_index + poses_3d[key].shape[0]


        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[key].shape[-1]))

        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[key].shape[-2], poses_3d[key].shape[-1]))
        self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d[key].shape[-2], poses_2d[key].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        if cameras is not None:
            self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.out_all = out_all

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def get_batch(self, seq_i, start_3d, end_3d, flip, reverse):
        subject,action,cam_index = seq_i
        seq_name = (subject,action,int(cam_index))
        start_2d = start_3d - self.pad - self.causal_shift
        end_2d = end_3d + self.pad - self.causal_shift

        seq_2d = self.poses_2d[seq_name].copy()
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            self.batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
        else:
            self.batch_2d = seq_2d[low_2d:high_2d]

        if flip:
            self.batch_2d[ :, :, 0] *= -1
            self.batch_2d[ :, self.kps_left + self.kps_right] = self.batch_2d[ :,
                                                                  self.kps_right + self.kps_left]
        if reverse:
            self.batch_2d = self.batch_2d[::-1].copy()

        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_name].copy()
            if self.out_all:
                low_3d = low_2d
                high_3d = high_2d
                pad_left_3d = pad_left_2d
                pad_right_3d = pad_right_2d
            else:
                low_3d = max(start_3d, 0)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left_3d = low_3d - start_3d
                pad_right_3d = end_3d - high_3d
            if pad_left_3d != 0 or pad_right_3d != 0:
                self.batch_3d = np.pad(seq_3d[low_3d:high_3d],
                                          ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
            else:
                self.batch_3d = seq_3d[low_3d:high_3d]

            if flip:
                self.batch_3d[ :, :, 0] *= -1
                self.batch_3d[ :, self.joints_left + self.joints_right] = \
                    self.batch_3d[ :, self.joints_right + self.joints_left]
            if reverse:
                self.batch_3d = self.batch_3d[::-1].copy()

        if self.cameras is not None:
            self.batch_cam = self.cameras[seq_name].copy()
            if flip:
                self.batch_cam[ 2] *= -1
                self.batch_cam[ 7] *= -1

        if self.poses_3d is None and self.cameras is None:
            return None, None, self.batch_2d.copy(), action, subject, int(cam_index)
        elif self.poses_3d is not None and self.cameras is None:
            return np.zeros(9), self.batch_3d.copy(), self.batch_2d.copy(),action, subject, int(cam_index)
        elif self.poses_3d is None:
            return self.batch_cam, None, self.batch_2d.copy(),action, subject, int(cam_index)
        else:
            return self.batch_cam, self.batch_3d.copy(), self.batch_2d.copy(),action, subject, int(cam_index)





            

