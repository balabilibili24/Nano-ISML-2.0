# coding:utf8
import warnings
import torch as t


class DefaultConfig(object):
    mark = 'a-1'

    env = 'default'  # visdom 环
    # 境
    vis_port =8097 # visdom 端口
    model = 'UNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = 'C:/Users/ZMS/Desktop/Confocal/' # 训练集存放路径
    test_data_root = train_data_root   # 测试集存放路径
    load_vessel_model_path = r"C:\Users\ZMS\Desktop\quanzhong\0.pth" # 绿色0 红色1
    load_FTn_model_path = r"C:\Users\ZMS\Desktop\quanzhong\1.pth"  # 绿色0 红色1
                                                  #weight---------3LL模型
                                                  #quanzhong------通用模型

    FTn_images_name = '_c1.png'
    Vessel_images_name = '_c2.png'
    Magenta_images_name = '_c3.png'
    Nuclei_images_name = '_c4.png'


    batch_size = 4    # batch size
    use_gpu = True    # user GPU or not
    num_workers = 8  # how many workers for loading data
    print_freq = 20   # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = None

    max_epoch = 100
    lr = 0.01 * 5  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数

    device_ids = [0]

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        opt.device =t.device('cuda') if opt.use_gpu else t.device('cpu')


        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()
