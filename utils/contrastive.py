import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def semi_loss(z1: torch.Tensor, z2: torch.Tensor, T):
    f = lambda x: torch.exp(x / T)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(
        between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
    )


def whole_batched_semi_loss(z1: torch.Tensor, z2: torch.Tensor, batch_size: int, T):
    # Space complexity: O(BN) (semi_loss: O(N^2))
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / T)
    indices = torch.arange(0, num_nodes).to(device)
    losses = []
    for i in range(num_batches):
        mask = indices[i * batch_size : (i + 1) * batch_size]
        refl_sim = f(sim(z1[mask], z1))  # [B, N]
        between_sim = f(sim(z1[mask], z2))  # [B, N]

        losses.append(
            -torch.log(
                between_sim[:, i * batch_size : (i + 1) * batch_size].diag()
                / (
                    refl_sim.sum(1)
                    + between_sim.sum(1)
                    - refl_sim[:, i * batch_size : (i + 1) * batch_size].diag()
                )
            )
        )


def batched_semi_loss(z1: torch.Tensor, z2: torch.Tensor, batch_size: int, T):
    # Space complexity: O(BN) (semi_loss: O(N^2))
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / T)
    indices = np.arange(0, num_nodes)
    np.random.shuffle(indices)
    i = 0
    mask = indices[i * batch_size : (i + 1) * batch_size]
    refl_sim = f(sim(z1[mask], z1))  # [B, N]
    between_sim = f(sim(z1[mask], z2))  # [B, N]
    loss = -torch.log(
        between_sim[:, i * batch_size : (i + 1) * batch_size].diag()
        / (
            refl_sim.sum(1)
            + between_sim.sum(1)
            - refl_sim[:, i * batch_size : (i + 1) * batch_size].diag()
        )
    )

    return loss


def com_semi_loss(z1: torch.Tensor, z2: torch.Tensor, T, com_nodes1, com_nodes2):
    f = lambda x: torch.exp(x / T)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(
        between_sim[com_nodes1, com_nodes2]
        / (
            refl_sim.sum(1)[com_nodes1]
            + between_sim.sum(1)[com_nodes1]
            - refl_sim.diag()[com_nodes1]
        )
    )


class SimCLRTau:
    def __init__(
        self,
        args,
    ):
        super(SimCLRTau).__init__()
        self.proj1 = nn.Linear(args.hidden, args.proj)
        self.proj2 = nn.Linear(args.hidden, args.proj)
        self.tau = nn.Linear(args.proj, 1)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, T, com_nodes1, com_nodes2):
        z1 = self.proj1(z1)
        z2 = self.proj2(z2)

        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(sim(z1, z1))
        between_sim = f(sim(z1, z2))

        return -torch.log(
            between_sim[com_nodes1, com_nodes2]
            / (
                refl_sim.sum(1)[com_nodes1]
                + between_sim.sum(1)[com_nodes1]
                - refl_sim.diag()[com_nodes1]
            )
        )


def contrastive_loss_node(x1, x2, args, com_nodes=None):
    T = args.t
    # if args.dname in ["yelp", "coauthor_dblp", "walmart-trips-100"]:
    #     batch_size=1024
    # else:
    #     batch_size = None
    batch_size = None
    if com_nodes is None:
        if batch_size is None:
            l1 = semi_loss(x1, x2, T)
            l2 = semi_loss(x2, x1, T)
        else:
            l1 = batched_semi_loss(x1, x2, batch_size, T)
            l2 = batched_semi_loss(x2, x1, batch_size, T)
    else:
        l1 = com_semi_loss(x1, x2, T, com_nodes[0], com_nodes[1])
        l2 = com_semi_loss(x2, x1, T, com_nodes[1], com_nodes[0])
    ret = (l1 + l2) * 0.5
    ret = ret.mean()

    return ret


def semi_loss_JSD(z1: torch.Tensor, z2: torch.Tensor):
    # f = lambda x: torch.exp(x / T)

    refl_sim = sim(z1, z1)
    between_sim = sim(z1, z2)
    N = refl_sim.shape[0]
    pos_score = (np.log(2) - F.softplus(-between_sim.diag())).mean()
    neg_score_1 = F.softplus(-refl_sim) + refl_sim - np.log(2)
    neg_score_1 = torch.sum(neg_score_1) - torch.sum(neg_score_1.diag())
    neg_score_2 = torch.sum(F.softplus(-between_sim) + between_sim - np.log(2))
    neg_score = (neg_score_1 + neg_score_2) / (N * (2 * N - 1))
    return neg_score - pos_score


def contrastive_loss_node_JSD(x1, x2, args, com_nodes=None):
    T = args.t
    # if args.dname in ["yelp", "coauthor_dblp", "walmart-trips-100"]:
    #     batch_size=1024
    # else:
    #     batch_size = None
    batch_size = None
    if com_nodes is None:
        if batch_size is None:
            l1 = semi_loss_JSD(x1, x2)
            l2 = semi_loss_JSD(x2, x1)
        else:
            l1 = batched_semi_loss(x1, x2, batch_size, T)
            l2 = batched_semi_loss(x2, x1, batch_size, T)
    else:
        l1 = com_semi_loss(x1, x2, T, com_nodes[0], com_nodes[1])
        l2 = com_semi_loss(x2, x1, T, com_nodes[1], com_nodes[0])
    ret = (l1 + l2) * 0.5
    return ret


def semi_loss_TM(z1: torch.Tensor, z2: torch.Tensor):
    # f = lambda x: torch.exp(x / T)
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    eps = 1.0
    N = z1.shape[0]
    pdist = nn.PairwiseDistance(p=2)
    pos_score = pdist(z1, z2).mean()
    neg_score_1 = torch.cdist(z1, z1, p=2)
    neg_score_2 = torch.cdist(z1, z2, p=2)
    neg_score_1 = torch.sum(neg_score_1) - torch.sum(neg_score_1.diag())
    neg_score_2 = torch.sum(neg_score_2)
    neg_score = (neg_score_1 + neg_score_2) / (N * (2 * N - 1))
    return torch.max(pos_score - neg_score + eps, 0)[0]


def contrastive_loss_node_TM(x1, x2, args, com_nodes=None):
    T = args.t
    # if args.dname in ["yelp", "coauthor_dblp", "walmart-trips-100"]:
    #     batch_size=1024
    # else:
    #     batch_size = None
    batch_size = None
    if com_nodes is None:
        if batch_size is None:
            l1 = semi_loss_TM(x1, x2)
            l2 = semi_loss_TM(x2, x1)
        else:
            l1 = batched_semi_loss(x1, x2, batch_size, T)
            l2 = batched_semi_loss(x2, x1, batch_size, T)
    else:
        l1 = com_semi_loss(x1, x2, T, com_nodes[0], com_nodes[1])
        l2 = com_semi_loss(x2, x1, T, com_nodes[1], com_nodes[0])
    ret = (l1 + l2) * 0.5
    return ret


def sim_d(z1: torch.Tensor, z2: torch.Tensor):
    # z1 = F.normalize(z1)
    # z2 = F.normalize(z2)
    return torch.sqrt(torch.sum(torch.pow(z1 - z2, 2), 1))


def calculate_distance(z1: torch.Tensor, z2: torch.Tensor):
    num_nodes = z1.size(0)
    refl_sim = 0
    for i in range(num_nodes):
        refl_sim += (
            torch.sum(sim_d(z1[i : i + 1], z1))
            - torch.squeeze(sim_d(z1[i : i + 1], z1[i : i + 1]))
        ) / (num_nodes - 1)
    refl_sim = refl_sim / (num_nodes)
    between_sim = torch.sum(sim_d(z1, z2)) / num_nodes
    # print(refl_sim, between_sim)


def create_hypersubgraph(data, args):
    sub_size = args.sub_size
    node_size = int(data.n_x[0].item())
    # hyperedge_size = int(data.num_hyperedges[0].item())
    hyperedge_size = int(data.num_hyperedges)
    sample_nodes = np.random.permutation(node_size)[:sub_size]
    sample_nodes = list(np.sort(sample_nodes))
    edge_index = data.edge_index
    device = edge_index.device
    sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
        sample_nodes, 1, edge_index, relabel_nodes=False, flow="target_to_source"
    )
    sub_nodes, sorted_idx = torch.sort(sub_nodes)
    # relabel
    node_idx = torch.zeros(
        2 * node_size + hyperedge_size, dtype=torch.long, device=device
    )
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=device)
    sub_edge_index = node_idx[sub_edge_index]
    x = data.x[sample_nodes]
    data_sub = Data(x=x, edge_index=sub_edge_index)
    data_sub.n_x = torch.tensor([sub_size])
    data_sub.num_hyperedges = torch.tensor([sub_nodes.size(0) - 2 * sub_size])
    data_sub.norm = 0
    data_sub.totedges = torch.tensor(sub_nodes.size(0) - sub_size)
    data_sub.num_ori_edge = sub_edge_index.shape[1] - sub_size
    return data_sub


def pair_cosine_similarity(x, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps)


def nt_xent(x, t=0.5):
    # print("device of x is {}".format(x.device))
    x = pair_cosine_similarity(x)
    x = torch.exp(x / t)
    idx = torch.arange(x.size()[0])
    # Put positive pairs on the diagonal
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    # subtract the similarity of 1 from the numerator
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))
    return -torch.log(x).mean()


def PGD_contrastive(
    model,
    data,
    eps=8.0 / 255.0,
    alpha=2.0 / 255.0,
    iters=1,
    singleImg=False,
    feature_gene=None,
    sameBN=False,
):
    # init
    inputs = data.x
    delta = torch.rand_like(inputs) * eps * 2 - eps
    delta = torch.nn.Parameter(delta)

    if singleImg:
        # project half of the delta to be zero
        idx = [i for i in range(1, delta.data.shape[0], 2)]
        delta.data[idx] = torch.clamp(delta.data[idx], min=0, max=0)

    for i in range(iters):
        if feature_gene is None:
            if sameBN:

                features = model.eval()(inputs + delta, "normal")
            else:
                data.x = inputs + delta
                features = model.eval()(data)
        else:
            features = feature_gene(model, inputs + delta, "eval")

        model.zero_grad()
        loss = nt_xent(features)
        loss.backward()
        # print("loss is {}".format(loss))

        delta.data = delta.data + alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(inputs + delta.data, min=0, max=1) - inputs

        if singleImg:
            # project half of the delta to be zero
            idx = [i for i in range(1, delta.data.shape[0], 2)]
            delta.data[idx] = torch.clamp(delta.data[idx], min=0, max=0)

    # data.x = (inputs + delta).detach()
    # return data.detach()
    return (inputs + delta).detach()
