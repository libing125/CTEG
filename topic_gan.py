from discrminator import Discriminator
import tensorflow as tf
from config import Config
from dataloader import *
import time
from generator import Generator

if __name__ == '__main__':

    log_file = "pretrain_log_essay_concept.txt"
    # set random seed for reproduce
    tf.set_random_seed(88)
    np.random.seed(88)

    config_g = Config().generator_config_zhihu
    config_d = Config().discriminator_config_zhihu
    training_config = Config().training_config_zhihu

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # load vocab
    vocab_dict = np.load(training_config["word_dict"]).item()
    idx2word = {v: k for k, v in vocab_dict.items()}
    print(len(vocab_dict))
    config_g["vocab_dict"] = vocab_dict
    config_g["pretrain_wv"] = np.load(training_config["pretrain_wv"])
    assert config_g["embedding_size"] == config_g["pretrain_wv"].shape[1]

    G = Generator(config_g)
    G.build_placeholder()
    G.build_graph()

    # prepare dataset loader
    si, sl, slbl, ti, tl, train_mem = load_npy(Config().train_data_path_zhihu)
    g_pre_dataloader = GenDataLoader(config_g["batch_size"], si, sl, ti, tl, max_len=120, memory=train_mem)
    g_adv_dataloader = GenDataLoader(config_g["batch_size"], si, sl, ti, tl, max_len=120, source_label=slbl,
                                     memory=train_mem)

    si_tst, sl_tst, slbl_tst, ti_tst, tl_tst, tst_mem = load_npy(Config().test_data_path_zhihu)
    g_test_dataloader = GenDataLoader(config_g["batch_size"], si_tst, sl_tst, ti_tst, tl_tst,
                                      max_len=config_g["max_len"], source_label=slbl_tst, memory=tst_mem)

    g_test_dataloader.create_batch()

    si_val, sl_val, slbl_val, ti_val, tl_val, val_mem = load_npy(Config().val_data_path_zhihu)
    g_val_dataloader = GenDataLoader(config_g["batch_size"], si_val, sl_val, ti_val, tl_val,
                                     max_len=config_g["max_len"], source_label=slbl_val, memory=val_mem)
    g_val_dataloader.create_batch()

    sess = tf.Session(config=tf_config)

    saver_g = tf.train.Saver()
    g_pre_dataloader.create_batch()
    sess.run(tf.global_variables_initializer())
    ############################### Pre-train ########################################
    total_step = 0
    best_bleu = 0.0
    print("Start pre-training generator")
    for e in range(1, training_config["pre_gen_epoch"] + 1):
        avg_loss = 0
        for _ in range(g_pre_dataloader.num_batch):
            total_step += 1
            batch = g_pre_dataloader.next_batch()
            pre_g_loss = G.run_pretrain_step(sess, batch)
            avg_loss += pre_g_loss
            if e >= 10 and total_step % 500 == 0:
                bleu = G.evaluate(sess, g_test_dataloader, idx2word)
                if bleu > best_bleu:
                    with open(log_file, "a+") as f:
                        f.write("step : %d bleu : %f \n " % (total_step, bleu))
                    best_bleu = bleu
                    saver_g.save(sess, training_config["generator_path"] + "generator-" + str(total_step))

        log_data = "epoch: %d average training loss: %.4f" % (e, avg_loss / g_pre_dataloader.num_batch)
        print(log_data)
        with open(log_file, "a+") as f:
            f.write(log_data + "\n")

    # pre-train discriminator
    d_pre_dataloader = DisDataLoader(sess, G, config_d["batch_size"], max_len=120, num_class=501,
                                     topic_input=si, topic_label=slbl, topic_len=sl, target_idx=ti, memory=train_mem)
    D = Discriminator(config_d)
    D.build_graph()
    # initialize
    d_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
    sess.run(tf.variables_initializer(d_var_list))
    d_path = tf.train.latest_checkpoint(training_config["discriminator_path"])
    print(d_path)
    saver_d = tf.train.Saver()

    print("Start pre-training discriminator")
    for e in range(training_config["pre_dis_epoch"]):
        # print("preparing data ....")
        t0 = time.time()
        d_pre_dataloader.prepare_data(training_config["generate_batch"])
        # print("data generated, time cost :  %.3f s" % (time.time() - t0))
        for rt in range(training_config["repeat_time"]):
            true_data_loss, true_data_acc, true_data_hl = 0, 0, 0
            fake_data_loss, fake_data_acc, fake_data_hl = 0, 0, 0
            p, r, f1 = 0, 0, 0
            for b in range(d_pre_dataloader.num_batch):
                batch_x, batch_y = d_pre_dataloader.next_batch()
                update, d_pre_loss, d_acc, d_hl, d_f1, d_p, d_r = D.run_train_epoch(sess, batch_x, batch_y,
                                                                                    fetch_f1=True)
                if b < d_pre_dataloader.num_batch // 2:
                    fake_data_loss += d_pre_loss
                    fake_data_acc += d_acc
                    fake_data_hl += d_hl
                else:
                    true_data_loss += d_pre_loss
                    true_data_acc += d_acc
                    true_data_hl += d_hl
                p += d_p
                r += d_r
                f1 += d_f1
            print("epoch %d :  \n true data loss: %.4f acc: %.3f HL: %.4f \n "
                  "fake data loss : %.4f acc: %.3f HL : %.4f\n" % (
                      e, true_data_loss / d_pre_dataloader.num_batch * 2,
                      true_data_acc / d_pre_dataloader.num_batch * 2,
                      true_data_hl / d_pre_dataloader.num_batch * 2,
                      fake_data_loss / d_pre_dataloader.num_batch * 2,
                      fake_data_acc / d_pre_dataloader.num_batch * 2,
                      fake_data_hl / d_pre_dataloader.num_batch * 2,
                  ))
            print("Micro-F1: %f  Precision: %f  Recall: %f" % (f1 / d_pre_dataloader.num_batch,
                                                               p / d_pre_dataloader.num_batch,
                                                               r / d_pre_dataloader.num_batch))

    saver_d.save(sess, training_config["discriminator_path"] + "after_pre_dis")
    ############################# adversarial training ###################################
    saver_adv = tf.train.Saver(max_to_keep=10)
    saver_best = tf.train.Saver(max_to_keep=1)
    g_adv_dataloader.create_batch()

    adv_step = 0
    best_path = training_config["best"]
    print("start adversarial training")
    best_bleu = 0
    for adv_e in range(training_config["adv_epoch"]):
        print("adversarial epoch %d start!" % (adv_e + 1))
        # training generator
        for g_e in range(training_config["adv_g_epoch"]):
            g_adv_dataloader.reset_pointer()
            for b_n in range(g_adv_dataloader.num_batch):
                adv_step += 1
                topic_idx, topic_len, target_idx, target_len, source_label, mem = g_adv_dataloader.next_batch()
                log_data = "epoch : %d    step:  %d \n" % (adv_e, b_n)
                samples = G.generate_essay(sess, topic_idx, topic_len, memory=mem, padding=True)
                rewards = G.get_reward(sess, samples, topic_idx, topic_len, rollout_num=training_config["rollout_num"],
                                       discriminator=D, source_label=source_label, memory=mem)
                log_data += "average reward: %f \n" % np.average(np.average(rewards, axis=1), axis=0)
                adv_batch = [topic_idx, topic_len, samples, mem]
                rewards_loss = G.run_adversarial_step(sess, adv_batch, rewards)
                log_data += "adversarial loss: %f\n" % rewards_loss

                # teacher forcing
                mle_batch = [topic_idx, topic_len, target_idx, target_len, mem]
                mle_loss = G.run_pretrain_step(sess, mle_batch)
                log_data += "mle loss: %f \n" % mle_loss

                with open("concept_mem_adv_log.txt", "a+") as f:
                    f.write(log_data)
                # evaluate every 100 step on validation dataset
                if adv_step % 100 == 0:
                    bleu = G.evaluate(sess, g_val_dataloader, idx2word)
                    with open("concept_mem_bleu.txt", "a+") as f:
                        f.write("adv step %d : bleu %f :\n" % (adv_step, bleu))
                    model_path = training_config["adv_path"] + "adv-" + str(adv_step)
                    print("saving to ", model_path)
                    saver_adv.save(sess, model_path)
                    if bleu > best_bleu:
                        best_bleu = bleu
                        saver_best.save(sess, best_path + ("%.3f" % (100 * best_bleu)))

        # discriminator epoch
        for d_e in range(training_config["adv_d_epoch"]):
            print("preparing data.....")
            d_pre_dataloader.prepare_data(training_config["generate_batch"])
            # print("data generated, time cost :  %.3f s" % (time.time() - t0))
            for rt in range(training_config["repeat_time"]):
                for _ in range(d_pre_dataloader.num_batch):
                    batch_x, batch_y = d_pre_dataloader.next_batch()
                    update, d_pre_loss, d_acc, d_hl = D.run_train_epoch(sess, batch_x, batch_y)
            print("pretrain discriminator loss: %.4f accuracy : %.3f hamming loss : %.4f" % (
                d_pre_loss, d_acc, d_hl))

    print("Training finished")
