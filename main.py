import os, os.path as osp
import torch
import time
import numpy as np
import copy
from tqdm import tqdm
from collections import defaultdict
import torch.nn as nn, torch.nn.functional as F
from utils import (
    parser_data,
    fix_seed,
    dataset_Hypergraph,
    ExtractV2E,
    Add_Self_Loops,
    expand_edge_index,
    norm_contruction,
    rand_train_test_idx,
    SetGNN,
    count_parameters,
    Logger,
    eval_acc,
    evaluate,
    evaluate_finetune,
    aug,
    create_hypersubgraph,
)

# from utils.dataLoader import dataset_Hypergraph
# from utils.preprocessing import ExtractV2E, Add_Self_Loops, expand_edge_index, norm_contruction


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def load_data(args):
    ### Load and preprocess data ###
    existing_dataset = [
        "20newsW100",
        "ModelNet40",
        "zoo",
        "NTU2012",
        "Mushroom",
        "coauthor_cora",
        "coauthor_dblp",
        "yelp",
        "amazon-reviews",
        "walmart-trips",
        "house-committees",
        "walmart-trips-100",
        "house-committees-100",
        "cora",
        "citeseer",
        "pubmed",
        "twitter",
    ]

    synthetic_list = [
        "amazon-reviews",
        "walmart-trips",
        "house-committees",
        "walmart-trips-100",
        "house-committees-100",
    ]
    if args.dname in existing_dataset:
        dname = args.dname
        f_noise = args.feature_noise
        p2raw = osp.join(args.data_dir, "AllSet_all_raw_data")
        if (f_noise is not None) and dname in synthetic_list:
            dataset = dataset_Hypergraph(name=dname, feature_noise=f_noise, p2raw=p2raw)
        else:
            if dname in ["cora", "citeseer", "pubmed"]:
                p2raw = osp.join(p2raw, "cocitation")
            elif dname in ["coauthor_cora", "coauthor_dblp"]:
                p2raw = osp.join(p2raw, "coauthorship")
            elif dname in ["yelp"]:
                p2raw = osp.join(p2raw, "yelp")
            elif dname in ["twitter"]:
                p2raw = osp.join(p2raw, "twitter")
            dataset = dataset_Hypergraph(
                name=dname,
                root=osp.join(args.data_dir, "pyg_data", "hypergraph_dataset_updated"),
                p2raw=p2raw,
            )
    data = dataset.data
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    if args.dname in [
        "yelp",
        "walmart-trips",
        "house-committees",
        "walmart-trips-100",
        "house-committees-100",
    ]:
        args.num_classes = len(data.y.unique())
        data.y = data.y - data.y.min()
    data.n_x = torch.tensor([data.x.shape[0]])
    data.num_hyperedges = torch.tensor([data.num_hyperedges])

    data = ExtractV2E(data)
    if args.add_self_loop:
        data = Add_Self_Loops(data)
    if args.exclude_self:
        data = expand_edge_index(data)
    data = norm_contruction(data, option=args.normtype)

    return data


if __name__ == "__main__":
    start = time.time()
    # data = "walmart-trips-100"
    data = "twitter"
    # data = "cora"
    args = parser_data(data)
    fix_seed(args.seed)

    # # Part 1: Load data
    data = load_data(args)

    #  Get Splits
    split_idx_lst = []
    for run in range(args.runs):  # how many runs
        split_idx = rand_train_test_idx(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop
        )  # train test split
        split_idx_lst.append(split_idx)  # the list of data splitting

    # # Part 2: Load model

    if args.method == "AllDeepSets":
        args.PMA = False
        args.aggregate = "add"
    elif args.method == "AllSetTransformer":
        pass
    else:
        raise ValueError("Method not implemented")
    if args.LearnMask:
        model = SetGNN(args, data.norm)
    else:
        model = SetGNN(args)

    # put things to device
    if args.cuda in [0, 1, 2, 3]:
        device = torch.device(
            "cuda:" + str(args.cuda) if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device("cpu")

    model = model.to(device)
    data = data.to(device)
    data_pre = copy.deepcopy(data)
    num_params = count_parameters(model)

    # # Part 3: Main. Training + Evaluation

    logger = Logger(args.runs, args)

    criterion = nn.NLLLoss()
    eval_func = eval_acc

    model.train()

    ### Training Loop ###
    he_index = defaultdict(list)
    edge_index = data.edge_index.cpu().numpy()
    for i, he in enumerate(edge_index[1]):
        he_index[he].append(i)
    runtime_list = []
    for run in tqdm(range(args.runs)):
        # for run in range(args.runs):
        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx["train"].to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd
        )
        best_val = float("-inf")
        # hyperedge_idx = [i for i in range(data.n_x,data.n_x+data.num_hyperedges)]
        for epoch in range(args.epochs):
            #         Training part
            model.train()
            optimizer.zero_grad()
            # cl loss
            if args.m_l:

                if data_pre.n_x <= args.sub_size:
                    data_sub = data_pre
                else:
                    data_sub = create_hypersubgraph(data_pre, args)  ###

                data_sub1 = copy.deepcopy(data_sub)
                data_sub2 = copy.deepcopy(data_sub)
                if args.aug1 == "subgraph" or args.aug1 == "drop":
                    node_num = data_sub1.x.shape[0]
                    n_walk = 128 if node_num > 128 else 8
                    start = torch.randint(
                        0, node_num, size=(n_walk,), dtype=torch.long
                    ).to(device)
                    data_sub1 = data_sub1.to(device)
                    data_aug1, nodes1, hyperedge1 = aug(
                        data_sub1, args.aug1, args, start
                    )

                else:
                    data_sub1 = data_sub1.to(device)
                    cidx = data_sub1.edge_index[1].min()
                    data_sub1.edge_index[1] -= cidx
                    # must starts from zero
                    data_sub1 = data_sub1.to(device)
                    # data_aug1 = aug(data_sub, args).to(device)
                    data_aug1, nodes1, hyperedge1 = aug(data_sub1, args.aug1, args)
                    data_aug1 = data_aug1.to(device)
                    # nodes1 = set([i for i in range(data_sub1.x.size()[0])])
                    data_aug1.edge_index[1] += cidx
                # hyperedge_idx1 = torch.tensor([i for i in range(data_aug1.x.shape[0], data_aug1.x.shape[0] + len(hyperedge1))]).to(device)
                hyperedge_idx1 = torch.tensor(
                    list(
                        range(
                            data_aug1.x.shape[0], data_aug1.x.shape[0] + len(hyperedge1)
                        )
                    )
                ).to(device)

                def edge_embed(idx, data_aug):

                    return data_aug.edge_index[0][
                        torch.where(data_aug.edge_index[1] == idx)[0]
                    ]

                data1_node2edge = [edge_embed(i, data_aug1) for i in hyperedge_idx1]

                data1_edgeidx_l, data1_node2edge_sample = [], []
                for i in range(len(data1_node2edge)):
                    if torch.numel(data1_node2edge[i]) > 0:
                        data1_edgeidx_l.append(i)
                        data1_node2edge_sample.append(data1_node2edge[i])

                data1_edgeidx = data_aug1.x.shape[0] + torch.tensor(data1_edgeidx_l).to(
                    device
                )

                pgd1 = torch.rand_like(data_aug1.x)
                data_aug_pgd1 = data_aug1.clone()
                data_aug_pgd1.x = data_aug1.x + pgd1
                # out1 = model.forward_cl(data_aug_pgd1)
                # out1,edge1 = model.forward_global_local(data_aug_pgd1,data1_node2edge,device)
                out1, edge1 = model.forward_global_local(
                    data_aug_pgd1, data1_node2edge_sample, data1_edgeidx, device
                )
                # out1, edge1 = model.forward_global_local(data_aug_pgd1, data1_node2edge,  device)
                # edge1 = edge1 + torch.rand_like(edge1)
                if args.aug2 == "subgraph" or args.aug2 == "drop":
                    node_num = data_sub2.x.shape[0]
                    n_walk = 128 if node_num > 128 else 8
                    start = torch.randint(
                        0, node_num, size=(n_walk,), dtype=torch.long
                    ).to(device)
                    data_sub2 = data_sub2.to(device)
                    data_aug2, nodes2, hyperedge2 = aug(
                        data_sub2, args.aug2, args, start
                    )
                else:
                    data_sub2 = data_sub2.to(device)
                    cidx = data_sub2.edge_index[1].min()
                    data_sub2.edge_index[1] -= cidx
                    # must starts from zero
                    data_sub2 = data_sub2.to(device)
                    # data_aug1 = aug(data_sub, args).to(device)
                    data_aug2, nodes2, hyperedge2 = aug(data_sub2, args.aug2, args)
                    data_aug2 = data_aug2.to(device)
                    # nodes2 = set([i for i in range(data_sub2.x.size()[0])])
                    data_aug2.edge_index[1] += cidx

                # data2_node2edge = list(map(lambda i: edge_embed(i, data_aug2), torch.unique(data_aug2.edge_index[1]).tolist()))
                # hyperedge_idx2 = [i for i in range(data_aug2.n_x, data_aug2.n_x + data_aug2.num_hyperedges)]
                # hyperedge_idx2 = torch.tensor([i for i in range(data_aug2.x.shape[0], data_aug2.x.shape[0] + len(hyperedge2))]).to(device)
                hyperedge_idx2 = torch.tensor(
                    list(
                        range(
                            data_aug2.x.shape[0], data_aug2.x.shape[0] + len(hyperedge2)
                        )
                    )
                ).to(device)
                # data2_node2edge = list(map(lambda i: edge_embed(i, data_aug2), hyperedge_idx))

                # data2_node2edge = list(map(lambda i: edge_embed(i, data_aug2), hyperedge_idx2))

                # s2 = list(torch.split(data_aug2.edge_index[1] == hyperedge_idx2.reshape(-1, 1), 1, dim=0))
                # data2_node2edge = list(map(lambda i: data_aug2.edge_index[0].reshape(1, -1)[i], s2))

                data2_node2edge = [edge_embed(i, data_aug2) for i in hyperedge_idx2]

                # data2_node2edge_sample = list(filter(lambda x: x.numel() > 0, data2_node2edge))
                # data2_edgeidx_l = [i for i in range(len(data2_node2edge)) if data2_node2edge[i].shape[0] != 0]

                data2_edgeidx_l, data2_node2edge_sample = [], []
                for i in range(len(data2_node2edge)):
                    if torch.numel(data2_node2edge[i]) > 0:
                        data2_edgeidx_l.append(i)
                        data2_node2edge_sample.append(data2_node2edge[i])

                # data2_edgeidx = torch.tensor(data2_edgeidx_l)
                # data2_edgeidx = data_aug2.x.shape[0] + torch.tensor(data2_edgeidx_l)
                data2_edgeidx = data_aug2.x.shape[0] + torch.tensor(data2_edgeidx_l)

                # data2_edgeidx_l = [i for i in range(len(data2_node2edge))]
                # data2_edgeidx = torch.tensor(data2_edgeidx_l)

                pgd2 = torch.rand_like(data_aug2.x)
                data_aug_pgd2 = data_aug2.clone()
                data_aug_pgd2.x = data_aug2.x + pgd2
                # out2 = model.forward_cl(data_aug_pgd2)
                # out2, edge2 = model.forward_global_local(data_aug_pgd2,data2)
                # out2, edge2 = model.forward_global_local(data_aug_pgd2, data2_node2edge, device)
                out2, edge2 = model.forward_global_local(
                    data_aug_pgd2, data2_node2edge_sample, data2_edgeidx, device
                )
                # out2, edge2 = model.forward_global_local(data_aug_pgd2, data2_node2edge,  device)
                # edge2 = edge2 + torch.rand_like(edge2)

                # if args.aug1 in ['drop','subgraph'] or args.aug2 in ['drop','subgraph']:
                com_sample = list(set(nodes1) & set(nodes2))
                dict_nodes1, dict_nodes2 = {
                    value: i for i, value in enumerate(nodes1)
                }, {value: i for i, value in enumerate(nodes2)}
                com_sample1, com_sample2 = [
                    dict_nodes1[value] for value in com_sample
                ], [dict_nodes2[value] for value in com_sample]
                # loss_cl = contrastive_loss_node(out1, out2, args, [com_sample1, com_sample2])
                loss_cl = model.get_loss(out1, out2, args.t, [com_sample1, com_sample2])

                com_edge = list(
                    set(data1_edgeidx.tolist()) & set(data2_edgeidx.tolist())
                )

                dict_edge1, dict_edge2 = {
                    value: i for i, value in enumerate(data1_edgeidx.tolist())
                }, {value: i for i, value in enumerate(data2_edgeidx.tolist())}

                com_edge1, com_edge2 = [dict_edge1[value] for value in com_edge], [
                    dict_edge2[value] for value in com_edge
                ]
                # loss_cl_gl = contrastive_loss_node(edge1, edge2, args, [com_edge1, com_edge2])
                loss_cl_gl = model.get_loss(
                    edge1, edge2, args.t, [com_edge1, com_edge2]
                )

                # print(loss_cl)
                # print(loss_cl_gl)

            else:
                loss_cl = 0
            # sup loss
            if args.linear:
                out = model.forward_finetune(data)
            else:
                out = model(data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], data.y[train_idx])
            loss += args.m_l * (loss_cl + loss_cl_gl)

            # loss += args.m_l * loss_cl
            # loss += args.m_l * loss_cl_h
            # if epoch==10:
            #     print(out_aug, data_aug1.edge_index[0][:100])
            #     print()
            #     # print(list(model.named_parameters()))
            #     exit()
            # torch.autograd.set_detect_anomaly(True)

            # loss.backward(create_graph=True)
            loss.backward()
            optimizer.step()
            #         if args.method == 'HNHN':
            #             scheduler.step()
            #         Evaluation part
            time2 = time.time()
            if args.linear:
                result = evaluate_finetune(model, data, split_idx, eval_func)
            else:
                result = evaluate(model, data, split_idx, eval_func)
            # logger.add_result(run, result[:3])
            logger.add_result(run, result[:6])
            # with open("{}_{}.csv".format(args.dname, args.m_l),"a+") as f:
            #     f.write(str(result[2])+"\n")

            if epoch % args.display_step == 0 and args.display_step > 0:
                print(
                    f"Epoch: {epoch:02d}, "
                    f"Train Loss: {loss:.4f}, "
                    f"Valid Loss: {result[7]:.4f}, "
                    f"Test  Loss: {result[8]:.4f}, "
                    f"Train Acc: {100 * result[0]:.2f}%, "
                    f"Valid Acc: {100 * result[1]:.2f}%, "
                    f"Test  Acc: {100 * result[2]:.2f}%, "
                    f"Train F1: {100 * result[3]:.2f}%, "
                    f"Valid F1: {100 * result[4]:.2f}%, "
                    f"Test  F1: {100 * result[5]:.2f}%, "
                )
        end_time = time.time()
        runtime_list.append(end_time - start_time)

        logger.print_statistics(run)
        end = time.time()
        mins = (end - start) / 60
        print("The running time is {}".format(mins))

    ### Save results ###
    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)

    best_val_acc, best_test_acc, test_f1 = logger.print_statistics()
    res_root = "hyperparameter_tunning/attack/"
    if not osp.isdir(res_root):
        os.makedirs(res_root)

    filename = f"{res_root}/{args.dname}_noise_{args.feature_noise}.csv"
    print(f"Saving results to {filename}")
    with open(filename, "a+") as write_obj:
        if args.p_lr:
            cur_line = f"attack_{args.attack}_{args.method}_{args.m_l}_{args.lr}_{args.wd}_{args.sub_size}_{args.heads}_aug_{args.aug}_ratio_{str(args.aug_ratio)}_t_{str(args.t)}_plr_{str(args.p_lr)}_pepoch_{str(args.p_epochs)}_player_{str(args.p_layer)}_phidden_{str(args.p_hidden)}_drop_{str(args.dropout)}_train_{str(args.train_prop)}"
            if args.add_e:
                cur_line += "_add_e"
        else:
            cur_line = f"attack_{args.attack}_{args.method}_{args.lr}_{args.wd}_{args.heads}_{str(args.dropout)}_train_{str(args.train_prop)}"
        cur_line += f",{args.aug1,args.aug2}"
        cur_line += f",{best_val_acc.mean():.3f} ± {best_val_acc.std():.3f}"
        cur_line += f",{best_test_acc.mean():.3f} ± {best_test_acc.std():.3f}"
        cur_line += f",{test_f1.mean():.3f} ± {test_f1.std():.3f}"
        cur_line += f",{num_params}, {avg_time:.2f}s, {std_time:.2f}s"
        cur_line += f",{avg_time // 60}min{(avg_time % 60):.2f}s"
        cur_line += f"\n"
        write_obj.write(cur_line)

    all_args_file = f"{res_root}/all_args_{args.dname}_attack_{args.attack}_noise_{args.feature_noise}.csv"
    with open(all_args_file, "a+") as f:
        f.write(str(args))
        f.write("\n")
    end = time.time()
    mins = (end - start) / 60
    # print("The running time is {}".format(mins))
    print("All done! Exit python code")
    quit()
