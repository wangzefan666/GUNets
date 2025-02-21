import os
import sys
import json
import time
import uuid
import pickle as pkl
from sklearn import metrics
import unfolder
from model import *
from os import path
from utils import load_big
from args import get_args


def set_seed(seed, cuda):
    """
    使实验结果可复现
    """
    print('Set Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True  # 是否返回确定的卷积算法
        torch.backends.cudnn.benchmark = False  # 是否搜索最合适的卷积实现算法，实现网络的加速


def main():
    args = get_args()
    root_path = os.getcwd() + '/'

    # Load Data
    t0 = time.time()
    adj, features, labels, tr_idx, va_idx, ts_idx = load_big(prefix=args.dataset)
    num_feature = features.shape[1]
    num_nodes = adj.shape[0]
    if args.if_multi_label:
        num_classes = labels.shape[1]
    else:
        if labels.shape[1] > 1:
            labels = labels.argmax(1)
        num_classes = int(np.max(labels)) + 1

    unfolder.init(adj, features, set(tr_idx), args.samp_pare_num, args.samp_num, args.samp_times, args.un_layer,
                  max_degree=args.max_degree,
                  max_samp_nei=args.max_samp_nei, if_normalized=args.if_normalized,
                  degree_normalized=args.degree_normalized,
                  if_self_loop=args.if_self_loop, if_bagging=args.if_sampling, if_sort=args.if_sort,
                  weight=args.weight, n_jobs=args.n_jobs, seed=args.pre_seed)

    appendix = ''
    if args.if_normalized:
        appendix += '-norm'
    if args.degree_normalized:
        appendix += '-degnorm'
    if args.if_self_loop:
        appendix += '-self'
    if args.if_sort:
        appendix += '-sort'
    appendix += '-degree' + str(args.max_degree)
    appendix += '-tr'
    pre_path = root_path + 'pre/'
    if not os.path.exists(pre_path):
        os.makedirs(pre_path)
    if args.if_sampling:
        emb_save_file = 'pre/{}-la{}-st{}-sp{}-sn{}-seed{}.pkl'.format(
            args.dataset, args.un_layer, args.samp_times, args.samp_pare_num, args.samp_num,
            args.pre_seed, appendix)
    else:
        emb_save_file = 'pre/{}-naive-la{}-weight{}{}.pkl'.format(args.dataset, args.un_layer, args.weight, appendix)

    print('-' * 60)

    if len(args.pre_load) > 0:
        emb_save_file = args.pre_load
    print('EMB File Name:%s' % emb_save_file)
    if path.exists(emb_save_file) and not args.recompute:
        print('Load Previous Node Features')
        with open(emb_save_file, 'rb') as f:
            d = pkl.load(f)
        tr_embs = d['tr']
        va_embs = d['va']
        ts_embs = d['ts']
        pre_cost = d['cost']
    else:
        print('Computing Node Features')
        tr_embs = unfolder.get_batch_emb(tr_idx)
        va_embs = unfolder.get_batch_emb(va_idx)
        ts_embs = unfolder.get_batch_emb(ts_idx)
        t1 = time.time()
        pre_cost = t1 - t0
        d = dict()
        d['tr'] = tr_embs
        d['va'] = va_embs
        d['ts'] = ts_embs
        d['cost'] = pre_cost
        with open(emb_save_file, 'wb') as f:
            pkl.dump(d, f, protocol=4)

    X_tr = torch.from_numpy(tr_embs).float().to(args.device)
    X_va = torch.from_numpy(va_embs).float().to(args.device)
    X_ts = torch.from_numpy(ts_embs).float().to(args.device)
    if args.if_multi_label:
        Y_tr = torch.from_numpy(labels[tr_idx]).float().to(args.device)
        Y_va = torch.from_numpy(labels[va_idx]).float().to(args.device)
        Y_ts = torch.from_numpy(labels[ts_idx]).float().to(args.device)
    else:
        Y_tr = torch.from_numpy(labels[tr_idx]).long().squeeze(dim=-1).to(args.device)
        Y_va = torch.from_numpy(labels[va_idx]).long().squeeze(dim=-1).to(args.device)
        Y_ts = torch.from_numpy(labels[ts_idx]).long().squeeze(dim=-1).to(args.device)

    print('Shape of Tensor', X_tr.shape)

    print('IF_MULTI_label', args.if_multi_label)
    if args.if_multi_label:
        loss_func = nn.BCEWithLogitsLoss().to(args.device)
    else:
        loss_func = nn.CrossEntropyLoss().to(args.device)

    def evaluate(net, X, Y, batch_size):
        with torch.no_grad():
            out_list = []
            for begin in range(0, X.shape[0], batch_size):
                X_batch = X[begin:min(begin + batch_size, X.shape[0]), :, :]
                out = net(X_batch)
                out_list.append(out)
        out = torch.cat(out_list, dim=0)
        loss = float(loss_func(out, Y).data.cpu().numpy())
        score = score_func(out, Y)
        return loss, score

    def score_func(out, Y):
        y_true = Y.data.cpu().numpy()
        if args.if_multi_label:
            y_pred = torch.sigmoid(out).data.cpu().numpy()
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0
            y_pred = y_pred.astype(np.int)
            y_true = y_true.astype(np.int)
            f1_mic = metrics.f1_score(y_true, y_pred, average="micro"),
            # f1_mac = metrics.f1_score(y_true, y_pred, average="macro")
            score = f1_mic[0]
        else:
            y_pred = out.argmax(1)
            score = metrics.accuracy_score(y_true, y_pred.data.cpu().numpy())
        return score

    result_list = []
    time_plot_main_list = []
    for run_count in range(args.run_times):
        t0 = time.time()
        set_seed(args.train_seed + run_count, args.cuda)
        gun = GUN(num_feature, num_classes, emb_size=args.emb_size, un_layer=args.un_layer,
                  if_trans_bn=args.if_trans_bn, trans_act=args.trans_act, if_trans_share=args.if_trans_share,
                  if_bn_share=args.if_bn_share, if_trans_bias=args.if_bias, trans_init=args.trans_init,
                  mlp_act=args.mlp_act, mlp_size=args.mlp_size, mlp_layer=args.mlp_layer, if_mlp_bn=args.if_mlp_bn,
                  mlp_init=args.mlp_init, bn_mom=args.bn_mom, drop_rate=args.drop_rate, device=args.device)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, gun.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        if args.if_output:
            print(gun)
        uuid_code = str(uuid.uuid4())[:4]
        save_path = root_path + 'model_save/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = save_path + args.dataset + uuid_code
        out_path = root_path + 'out_res/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        patience_count = 0
        va_acc_max = 0
        ts_acc = 0
        batch = 0
        time_list = []
        time_plot_list = []
        early_stop_flag = False
        train_time = 0
        pure_train_time = 0
        his_file = ''
        batch_size = args.warm_batch_size
        for epoch in range(50000):
            if early_stop_flag:
                break
            random.shuffle(tr_idx)
            t_epoch_begin = time.time()
            epoch_train_cost = []
            epoch_val_cost = []
            for ind in range(0, len(tr_idx), batch_size):
                t_train_begin = time.time()
                if batch == args.warm_batch_num:
                    batch_size = args.batch_size
                    print('*' * 25, 'Updating BatchSize to %d' % (batch_size), '*' * 25)
                    batch += 1
                    break
                batch += 1
                end_ind = min(ind + args.batch_size, len(tr_idx))
                if end_ind - ind == 0:
                    continue
                batch_nodes = list(range(ind, end_ind))
                gun.train()
                start_time = time.time()
                optimizer.zero_grad()
                batch_x = X_tr[batch_nodes]
                batch_y = Y_tr[batch_nodes]
                tr_out = gun(batch_x)
                loss = loss_func(tr_out, batch_y)
                loss.backward()
                optimizer.step()
                # tr_acc = score_func(tr_out, batch_y)
                t_train_end = time.time()
                epoch_train_cost.append(t_train_end - t_train_begin)
                train_time += t_train_end - t_train_begin
                if args.if_output:
                    print('Epo%d-%d(%d/%d) loss:%.4f|Train time:%.2f' % (
                        epoch + 1, batch + 1, patience_count, args.patience, loss.data, train_time))
                if batch % args.output_batch == 0:
                    tr_acc = score_func(tr_out, batch_y)
                    gun.eval()
                    t_val_begin = time.time()
                    va_acc = evaluate(gun, X_va, Y_va, 2 * args.batch_size)[1]
                    time_plot_list.append([epoch, train_time, va_acc])
                    if va_acc > va_acc_max:
                        va_acc_max = va_acc
                        torch.save(gun.state_dict(), save_file)
                        patience_count = 0
                        gun.eval()
                        ts_acc = evaluate(gun, X_ts, Y_ts, 2 * args.batch_size)[1]
                    else:
                        patience_count += 1
                        if patience_count > args.patience:
                            early_stop_flag = True
                            break
                    t_val_end = time.time()
                    epoch_val_cost.append(t_val_end - t_val_begin)
                    if args.if_output:
                        print('Epo%d-%d(%d/%d) loss:%.4f|Tr:%.4f|Va:%.4f|BestVa:%.4f Ts:%.4f|Train time:%.2f' % (
                            epoch + 1, batch + 1, patience_count, args.patience, loss.data, tr_acc, va_acc,
                            va_acc_max, ts_acc, train_time))
            t_epoch_end = time.time()
            time_list.append([
                t_epoch_end - t_epoch_begin,
                np.sum(epoch_train_cost),
                np.sum(epoch_val_cost)
            ])
            if args.if_output:
                print('-' * 5, 'Train %d epoch in %.2f[%.2f](%.2f) second' % (
                    epoch, time.time() - t0, train_time, time_list[-1][0]), '-' * 5)
        time_array = np.array(time_list)
        t1 = time.time()
        gun.load_state_dict(torch.load(save_file))
        gun.eval()
        ts_acc = evaluate(gun, X_ts, Y_ts, 2 * args.batch_size)[1]
        running_cost = t1 - t0
        result = [
            batch,
            ts_acc,
            running_cost,  # Total Running time
            np.mean(time_array[:, 0]),  # Per epoch running time
            np.mean(time_array[:, 1]),  # Per epoch training time
            np.mean(time_array[:, 2]),  # Per epoch val time
            train_time
        ]
        with open('./' + args.dataset + '_' + args.name + '_time.txt', 'w') as f:
            f.write('epoch, time_train, va_acc\n')
            for _ in time_plot_list:
                f.write('%d,%.4f,%.4f\n' % (_[0], _[1], _[2]))
        print('#' * 30, 'Running Time %d' % (run_count), '#' * 30)
        print('Final Batch:%d, TS_score:%.4f, PreCompute time:%.2f, Running time:%.2f, Training time:%.2f' % (
            result[0], result[1], pre_cost, result[2], result[6]))
        result_list.append(result)
        time_plot_main_list.append(time_plot_list)
        del gun
        torch.cuda.empty_cache()
        result_array = np.array(result_list)
        args.mean_ts_acc = np.mean(result_array[:, 1])
        args.std_ts_acc = np.std(result_array[:, 1])
        print('*' * 60)
        print('Mean TS_score:%.4f' % args.mean_ts_acc)
        print('STD TS_score:%.4f' % args.std_ts_acc)
        args.pre_computing_cost = pre_cost
        args.running_cost = np.mean(result_array[:, 2])
        args.total_time_cost = args.pre_computing_cost + args.running_cost
        args.epoch_run_cost = np.mean(result_array[:, 3])
        args.epoch_train_cost = np.mean(result_array[:, 4])
        args.epoch_val_cost = np.mean(result_array[:, 5])
        device_bak = args.device
        args.device = args.device.type
        args.result_list = result_list
        args.final_batch = np.mean(result_array[:, 0])
        args.cmd = ' '.join(['python'] + sys.argv)
        args.time_list = time_plot_main_list
        out_file = out_path + args.name + args.dataset + "-%.4f-%d-" % (args.mean_ts_acc, args.run_times) + uuid_code
        if run_count == args.run_times - 1:
            with open(out_file, "w") as f:
                json.dump(vars(args), f)
        args.device = device_bak


if __name__ == '__main__':
    sys.exit(main())
