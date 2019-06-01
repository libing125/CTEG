class Config(object):
    training_config_zhihu = {
        "pre_gen_epoch": 150,
        "pre_dis_epoch": 50,
        "generate_batch": 256,  # zhihu 256 x 256 ~ one epoch  essay  2048 x 256
        "repeat_time": 3,
        "adv_epoch": 20,
        "rollout_num": 16,
        "adv_g_epoch": 1,
        "adv_d_epoch": 1,
        # zhihu
        "generator_path": "./zhihu_model/no_mem/generator/",
        "discriminator_path": "./zhihu_model/no_mem/discriminator/",
        "adv_path": "./zhihu_model/no_mem/adversarial/",
        "best": "./zhihu_model/no_mem/best/",
        "classifier_path": "./zhihu_model/classifier/",
        "word_dict": "./data/word_dict_zhihu.npy",
        "pretrain_wv": "./data/wv_tencent.npy",
        "topic_list": "./data/topic_list_100.pkl"

    }

    train_data_path_zhihu = [
        # "si_train":
        "./data/train_src.npy",
        # "sl_train":
        "./data/train_src_len.npy",
        # "s_lbl_train":
        "./data/train_src_lbl_oh.npy",
        # "ti_train":
        "./data/train_tgt.npy",
        # "tl_train":
        "./data/train_tgt_len.npy",
        # memory
        "./data/train_mem_idx_120_concept.npy"
    ]

    test_data_path_zhihu = [
        # "si_train":
        "./data/tst.src.npy",
        # "sl_train":
        "./data/tst.src.len.npy",
        # "s_lbl_train":
        "./data/tst.src.lbl.oh.npy",
        # "ti_train":
        "./data/tst.tgt.npy",
        # "tl_train":
        "./data/tst.tgt.len.npy",
        # memory
        "./data/tst.mem.idx.120.concept.npy"
    ]

    val_data_path_zhihu = [
        # "si_train":
        "./data/val.src.npy",
        # "sl_train":
        "./data/val.src.len.npy",
        # "s_lbl_train":
        "./data/val.src.lbl.oh.npy",
        # "ti_train":
        "./data/val.tgt.npy",
        # "tl_train":
        "./data/val.tgt.len.npy",
        # memory
        "./data/val.mem.idx.120.concept.npy"
    ]

    discriminator_config_zhihu = {
        "max_len": 100,  # zhihu 100
        "vocab_size": 50004,
        "embedding_size": 32,
        "learning_rate": 1e-4,
        "l2_reg_lambda": 0.0,
        "batch_size": 256,
        "topic_num": 5,
        "n_class": 101,  # zhihu 101
        # random setting, may need fine-tune
        "filter_sizes": [1, 2, 3, 4, 5, 10, 20, 50, 100],
        "num_filters": [128, 256, 256, 256, 256, 128, 128, 128, 256],
        "label_smooth": 0.9
    }

    classifier_config_zhihu = {  # the classifier  LSTM RNN classifier
        "max_len": 100,  # zhihu 100 essay 120
        "vocab_size": 50004,
        "embedding_size": 200,  # use pretrain word embedding
        "learning_rate": 1e-3,  # accelerate training
        "l2_reg_lambda": 1e-4,
        "batch_size": 256,
        "topic_num": 5,
        "n_class": 101,  # essay 501 zhihu 101
        # random setting, may need fine-tune
        "filter_sizes": [1, 2, 3, 4, 5, 10, 20, 50, 100],
        "num_filters": [64, 128, 128, 128, 128, 64, 64, 64, 128],
        "label_smooth": 0.9,
        "pretrain_wv_path": "./data/wv_tencent.npy"
    }

    classifier_config_zhihu_cnn = {  # the classifier is not perform well
        "max_len": 100,  # zhihu 100 essay 120
        "vocab_size": 50004,
        "embedding_size": 200,  # use pretrain word embedding
        "learning_rate": 1e-3,  # accelerate training
        "l2_reg_lambda": 1e-3,
        "batch_size": 64,
        "topic_num": 5,
        "n_class": 101,  # essay 501 zhihu 101
        # random setting, may need fine-tune
        "filter_sizes": [3, 4, 5],
        "num_filters": [128, 128, 128],
        "label_smooth": 0.9,
        "pretrain_wv_path": "./data/wv_tencent.npy"

    }

    generator_config_zhihu = {
        "embedding_size": 200,  # tencent 200 dim
        "hidden_size": 512,
        "max_len": 100,
        "start_token": 0,
        "eos_token": 1,
        "batch_size": 64,
        "vocab_size": 50004,
        "grad_norm": 10,
        "topic_num": 5,
        "is_training": True,
        "keep_prob": .5,
        "norm_init": 0.05,
        "normal_std": 1,
        "learning_rate": 1e-3,
        "beam_width": 5,
        "mem_num": 120,
        "attention_size": 128
    }
