from pprint import pprint
from pathlib import Path


class Config:
    # > File paths
    proj_path = Path('/shared_folder/')
    db_path = proj_path / 'data/tianchi/fashionAI_key_points/'

    checkpoint_path = db_path / 'checkpoints/'
    pred_path = db_path / 'kp_predictions/'
    log_path = db_path / 'logs/'
    checkpoint_path.mkdir(exist_ok=True)
    pred_path.mkdir(exist_ok=True)
    log_path.mkdir(exist_ok=True)

    load_checkpoint_path = None
    load_checkpoint_path_2 = None

    # > Visdom
    # env = 'default'

    # > Model parameters
    # 256 pixels: SGD L1 loss starts from 1e-2, L2 loss starts from 1e-3
    # 512 pixels: SGD L1 loss starts from 1e-3, L2 loss starts from 1e-4
    lr = 1e-3
    max_epochs = 1000
    batch_size = 4
    num_workers = 10
    gpu = '0'
    visual = False

    model_dict = {
        'ensemble': ('CascadedPyramidResNet', 'CascadedPyramidSENet'),
        'cpn-resnet': 'CascadedPyramidResNet',
        'cpn-senet': 'CascadedPyramidSENet'
    }
    model = None

    # > FashionAI keypoints dataset
    kp_dict = {
        'blouse': [
            'neckline_left', 'neckline_right', 'center_front', 'shoulder_left',
            'shoulder_right', 'armpit_left', 'armpit_right', 'cuff_left_in',
            'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left',
            'top_hem_right'
        ],
        'outwear': [
            'neckline_left', 'neckline_right', 'shoulder_left',
            'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left',
            'waistline_right', 'cuff_left_in', 'cuff_left_out',
            'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right'
        ],
        'trousers': [
            'waistband_left', 'waistband_right', 'crotch', 'bottom_left_in',
            'bottom_left_out', 'bottom_right_in', 'bottom_right_out'
        ],
        'skirt':
        ['waistband_left', 'waistband_right', 'hemline_left', 'hemline_right'],
        'dress': [
            'neckline_left', 'neckline_right', 'center_front', 'shoulder_left',
            'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left',
            'waistline_right', 'cuff_left_in', 'cuff_left_out',
            'cuff_right_in', 'cuff_right_out', 'hemline_left', 'hemline_right'
        ]
    }

    category = None
    keypoints = None
    num_keypoints = None
    conjug = None
    datum = None  # TODO: I think this is only used for validation

    # > Image
    img_max_size = 512
    # TODO: Are these normalization stats?
    mu = 0.65
    sigma = 0.25

    # > Gaussian heatmap
    #
    # According to the author: "If sigma is large, it will undermine
    # the accuracy, obviously. And if sigma is too small, I found that the
    # model cannot distinguish left and right."
    hm_stride = 4
    hm_sigma = img_max_size / hm_stride / 16.
    hm_alpha = 100.

    # > LR scheduler
    lrschedule_dict = {'blouse' : [16, 26, 42],
                       'outwear' : [15, 20, 26],
                       'trousers' : [18, 25, 36],
                       'skirt' : [26, 32, 39],
                       'dress' : [30, 34, 31]}

    lrschedule = None

    def _parse(self, kwargs):
        """Update configuration with the given kwargs dict.

        Args:
            kwargs (dict): Key-value pairs from Google Fire.
        """
        state_dict = self._state_dict()

        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)

            setattr(self, k, v)

            if v in self.kp_dict:
                kpts = self.kp_dict[v]
                setattr(self, 'keypoints', kpts)
                setattr(self, 'num_keypoints', len(kpts))
                setattr(self, 'lrschedule', self.lrschedule_dict[v])

                conjug = []
                for i, key in enumerate(kpts):
                    if 'left' in key:
                        j = kpts.index(key.replace('left', 'right'))
                        conjug.append([i, j])

                setattr(self, 'conjug', conjug)

                # TODO: I think this is only used for validation
                if v in ['outwear', 'blouse', 'dress']:
                    datum = [kpts.index('armpit_left'), kpts.index('armpit_right')]
                elif v in ['trousers', 'skirt']:
                    datum = [kpts.index('waistband_left'), kpts.index('waistband_right')]

                setattr(self, 'datum', datum)

            if v in self.model_dict:
                setattr(self, 'model', self.model_dict[v])

        print('====== User config ========')
        user_dict = {k: v for k, v in self._state_dict().items() \
                     if k not in ['kp_dict', 'lrschedule_dict']}
        pprint(user_dict)
        print('========== end ============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

opt = Config()
