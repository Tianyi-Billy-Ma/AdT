import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter
from torch_geometric.typing import Adj, Size, OptTensor
from typing import Optional


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


class SimCLRTau(nn.Module):
    def __init__(self, args):
        super(SimCLRTau, self).__init__()
        if args.p_hidden > 0:
            pre_hidden = args.p_hidden
        else:
            pre_hidden = args.Classifier_hidden

        self.batch_size = args.batch_size
        self.args = args

        self.fc1 = nn.Linear(pre_hidden, 200)
        self.fc2 = nn.Linear(200, args.MLP_hidden)
        self.tau1 = args.t
        # self.tau2 = 0.1
        self.low = 0.1
        self.pre_grad = 0.0
        # self.tau = nn.Linear(args.MLP_hidden, 1)
        # nn.init.xavier_uniform_(self.tau)
        # self.reset_parameters()

    def project(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def uniform_loss(self, z: torch.Tensor, t: int = 2):
        return torch.pdist(z, p=2).pow(2).mul(-t).exp().mean().log()

    def momentum(
        self,
        x_start: float,
        z: torch.Tensor,
        step: float = 0.001,
        discount: float = 0.7,
    ):
        if x_start <= self.low:
            return x_start
        x = x_start
        grad = -self.uniform_loss(z).item()
        self.pre_grad = self.pre_grad * discount + 1 / grad
        x -= self.pre_grad * step

        # x -= grad * step
        return x

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, T, com_nodes1, com_nodes2):
        z1 = self.project(z1)
        z2 = self.project(z2)

        self.tau1 = self.momentum(self.tau1, z1)
        #  self.tau1 = 1

        f = lambda x: torch.exp(x / self.tau1)

        refl_sim = f(sim(z1, z1).cpu())
        between_sim = f(sim(z1, z2).cpu())

        # refl_sim = f(sim(z1, z1))
        # between_sim = f(sim(z1, z2))

        if self.args.cl_loss == "InfoNCE":
            return -torch.log(
                between_sim[com_nodes1, com_nodes2]
                / (
                    refl_sim.sum(1)[com_nodes1]
                    + between_sim.sum(1)[com_nodes1]
                    - refl_sim.diag()[com_nodes1]
                )
            ).cuda()
        elif self.args.cl_loss == "JSD":
            N = refl_sim.shape[0]
            pos_score = (np.log(2) - F.softplus(-between_sim.diag())).mean()
            neg_score_1 = F.softplus(-refl_sim) + refl_sim - np.log(2)
            neg_score_1 = torch.sum(neg_score_1) - torch.sum(neg_score_1.diag())
            neg_score_2 = torch.sum(F.softplus(-between_sim) + between_sim - np.log(2))
            neg_score = (neg_score_1 + neg_score_2) / (N * (2 * N - 1))
            return neg_score - pos_score

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


class SetGNN(nn.Module):
    def __init__(self, args, norm=None, sig=False):
        super(SetGNN, self).__init__()
        """
        args should contain the following:
        V_in_dim, V_enc_hid_dim, V_dec_hid_dim, V_out_dim, V_enc_num_layers, V_dec_num_layers
        E_in_dim, E_enc_hid_dim, E_dec_hid_dim, E_out_dim, E_enc_num_layers, E_dec_num_layers
        All_num_layers,dropout
        !!! V_in_dim should be the dimension of node features
        !!! E_out_dim should be the number of classes (for classification)
        """
        #         V_in_dim = V_dict['in_dim']
        #         V_enc_hid_dim = V_dict['enc_hid_dim']
        #         V_dec_hid_dim = V_dict['dec_hid_dim']
        #         V_out_dim = V_dict['out_dim']
        #         V_enc_num_layers = V_dict['enc_num_layers']
        #         V_dec_num_layers = V_dict['dec_num_layers']

        #         E_in_dim = E_dict['in_dim']
        #         E_enc_hid_dim = E_dict['enc_hid_dim']
        #         E_dec_hid_dim = E_dict['dec_hid_dim']
        #         E_out_dim = E_dict['out_dim']
        #         E_enc_num_layers = E_dict['enc_num_layers']
        #         E_dec_num_layers = E_dict['dec_num_layers']

        #         Now set all dropout the same, but can be different
        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.InputNorm = args.deepset_input_norm
        self.GPR = args.GPR
        self.LearnMask = args.LearnMask
        self.args = args
        self.sig = sig
        self.args = args
        #         Now define V2EConvs[i], V2EConvs[i] for ith layers
        #         Currently we assume there's no hyperedge features, which means V_out_dim = E_in_dim
        #         If there's hyperedge features, concat with Vpart decoder output features [V_feat||E_feat]
        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()

        if self.LearnMask:
            self.Importance = Parameter(torch.ones(norm.size()))

        if self.All_num_layers == 0:
            self.classifier = MLP(
                in_channels=args.num_features,
                hidden_channels=args.Classifier_hidden,
                out_channels=args.num_classes,
                num_layers=args.Classifier_num_layers,
                dropout=self.dropout,
                Normalization=self.NormLayer,
                InputNorm=False,
            )
        else:
            self.V2EConvs.append(
                HalfNLHconv(
                    in_dim=args.num_features,
                    hid_dim=args.MLP_hidden,
                    out_dim=args.MLP_hidden,
                    num_layers=args.MLP_num_layers,
                    dropout=self.dropout,
                    Normalization=self.NormLayer,
                    InputNorm=self.InputNorm,
                    heads=args.heads,
                    attention=args.PMA,
                )
            )
            self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            self.E2VConvs.append(
                HalfNLHconv(
                    in_dim=args.MLP_hidden,
                    hid_dim=args.MLP_hidden,
                    out_dim=args.MLP_hidden,
                    num_layers=args.MLP_num_layers,
                    dropout=self.dropout,
                    Normalization=self.NormLayer,
                    InputNorm=self.InputNorm,
                    heads=args.heads,
                    attention=args.PMA,
                )
            )
            self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            for _ in range(self.All_num_layers - 1):
                self.V2EConvs.append(
                    HalfNLHconv(
                        in_dim=args.MLP_hidden,
                        hid_dim=args.MLP_hidden,
                        out_dim=args.MLP_hidden,
                        num_layers=args.MLP_num_layers,
                        dropout=self.dropout,
                        Normalization=self.NormLayer,
                        InputNorm=self.InputNorm,
                        heads=args.heads,
                        attention=args.PMA,
                    )
                )
                self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
                self.E2VConvs.append(
                    HalfNLHconv(
                        in_dim=args.MLP_hidden,
                        hid_dim=args.MLP_hidden,
                        out_dim=args.MLP_hidden,
                        num_layers=args.MLP_num_layers,
                        dropout=self.dropout,
                        Normalization=self.NormLayer,
                        InputNorm=self.InputNorm,
                        heads=args.heads,
                        attention=args.PMA,
                    )
                )
                self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))

            if self.GPR:
                self.MLP = MLP(
                    in_channels=args.num_features,
                    hidden_channels=args.MLP_hidden,
                    out_channels=args.MLP_hidden,
                    num_layers=args.MLP_num_layers,
                    dropout=self.dropout,
                    Normalization=self.NormLayer,
                    InputNorm=False,
                )
                self.GPRweights = Linear(self.All_num_layers + 1, 1, bias=False)
                self.classifier = MLP(
                    in_channels=args.MLP_hidden,
                    hidden_channels=args.Classifier_hidden,
                    out_channels=args.num_classes,
                    num_layers=args.Classifier_num_layers,
                    dropout=self.dropout,
                    Normalization=self.NormLayer,
                    InputNorm=False,
                )
            else:
                self.classifier = MLP(
                    in_channels=args.MLP_hidden,
                    hidden_channels=args.Classifier_hidden,
                    out_channels=args.num_classes,
                    num_layers=args.Classifier_num_layers,
                    dropout=self.dropout,
                    Normalization=self.NormLayer,
                    InputNorm=False,
                )

            # pretrain
            if args.p_layer > 0:
                pre_layer = args.p_layer
            else:
                pre_layer = args.Classifier_num_layers
            if args.p_hidden > 0:
                pre_hidden = args.p_hidden
            else:
                pre_hidden = args.Classifier_hidden

            self.proj_head = MLP(
                in_channels=args.MLP_hidden,
                hidden_channels=pre_hidden,
                out_channels=pre_hidden,
                num_layers=pre_layer,
                dropout=self.dropout,
                Normalization=self.NormLayer,
                InputNorm=False,
            )
            # self.V2EConvsLast = HalfNLHconv(in_dim=pre_hidden,
            #                                  hid_dim=pre_hidden,
            #                                  out_dim=pre_hidden,
            #                                  num_layers=args.MLP_num_layers,
            #                                  dropout=self.dropout,
            #                                  Normalization=self.NormLayer,
            #                                  InputNorm=self.InputNorm,
            #                                  heads=args.heads,
            #                                  attention=args.PMA)

            self.linear = nn.Linear(args.MLP_hidden, args.num_classes)
            self.decoder = nn.Sequential(
                nn.Linear(args.MLP_hidden, args.MLP_hidden),
                nn.ReLU(),
                nn.Linear(args.MLP_hidden, 2),
            )
            self.edge = nn.Linear(args.MLP_hidden + pre_hidden, pre_hidden)
            self.simclr_tau = SimCLRTau(args)

    #         Now we simply use V_enc_hid=V_dec_hid=E_enc_hid=E_dec_hid
    #         However, in general this can be arbitrary.

    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()
        if self.GPR:
            self.MLP.reset_parameters()
            self.GPRweights.reset_parameters()
        if self.LearnMask:
            nn.init.ones_(self.Importance)

    def forward(self, data):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
        #             The data should contain the follows
        #             data.x: node features
        #             data.V2Eedge_index:  edge list (of size (2,|E|)) where
        #             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges

        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnMask:
            norm = self.Importance * norm
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            x = self.classifier(x)
        else:
            # if not self.sig:
            x = F.dropout(x, p=0.2, training=self.training)  # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr))
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.classifier(x)
        return x

    def forward_link(self, data):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
        #             The data should contain the follows
        #             data.x: node features
        #             data.V2Eedge_index:  edge list (of size (2,|E|)) where
        #             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges

        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnMask:
            norm = self.Importance * norm
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            x = self.classifier(x)
        else:
            # if not self.sig:
            x = F.dropout(x, p=0.2, training=self.training)  # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                x_he = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x_he, p=self.dropout, training=self.training)
                x_node = F.relu(
                    self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                )
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x_node, p=self.dropout, training=self.training)
        return x_he, x_node

    def forward_finetune(self, data):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
        #             The data should contain the follows
        #             data.x: node features
        #             data.V2Eedge_index:  edge list (of size (2,|E|)) where
        #             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges

        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnMask:
            norm = self.Importance * norm
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            x = self.linear(x)
        else:
            x = F.dropout(x, p=0.2, training=self.training)  # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr))
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.linear(x)

        return x

    def forward_embed(self, data):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
        #             The data should contain the follows
        #             data.x: node features
        #             data.V2Eedge_index:  edge list (of size (2,|E|)) where
        #             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges

        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnMask:
            norm = self.Importance * norm
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            x = self.linear(x)
        else:
            x = F.dropout(x, p=0.2, training=self.training)  # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr))
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def forward_cl(self, data, aug_weight=None):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
        #             The data should contain the follows
        #             data.x: node features
        #             data.V2Eedge_index:  edge list (of size (2,|E|)) where
        #             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges

        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnMask:
            norm = self.Importance * norm
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            x = self.proj_head(x)
        else:
            # if self.dropout:
            if self.args.aug != "none":
                x = F.dropout(x, p=0.2, training=self.training)  # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                # print(edge_index[0].unique())
                # print(x.shape)
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr, aug_weight))
                # print(x.shape)
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(
                    self.E2VConvs[i](
                        x, reversed_edge_index, norm, self.aggr, aug_weight
                    )
                )
                # print(x.shape)
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.proj_head(x)
        return x

    # def forward_global_local(self, data, node2edge,device,aug_weight=None):
    def forward_global_local(
        self, data, node2edge, sample_edge_idx, device, aug_weight=None
    ):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
        #             The data should contain the follows
        #             data.x: node features
        #             data.V2Eedge_index:  edge list (of size (2,|E|)) where
        #             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges

        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnMask:
            norm = self.Importance * norm
        # cidx = edge_index[1].min()
        cidx = x.shape[0]
        # edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            x = self.proj_head(x)
        else:
            # if self.dropout:
            if self.args.aug != "none":
                x = F.dropout(x, p=0.2, training=self.training)  # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                # print(edge_index[0].unique())
                # print(x.shape)
                h1 = F.relu(
                    self.V2EConvs[i](x, edge_index, norm, self.aggr, aug_weight)
                )
                # print(x.shape)
                #                 x = self.bnV2Es[i](x)
                h1_d = F.dropout(h1, p=self.dropout, training=self.training)
                h2 = F.relu(
                    self.E2VConvs[i](
                        h1_d, reversed_edge_index, norm, self.aggr, aug_weight
                    )
                )
                # print(x.shape)
                #                 x = self.bnE2Vs[i](x)
                h2_d = F.dropout(h2, p=self.dropout, training=self.training)
            x = self.proj_head(h2_d)

            # h1 = h1[x.shape[0]:x.shape[0]+len(node2edge)]
            h1 = h1[sample_edge_idx]

            def edge_embed(idx):
                if self.args.edge == "sum":

                    return sum(x[idx]).reshape(1, -1)

                elif self.args.edge == "mean":
                    return torch.mean(x[idx], dim=0).reshape(1, -1)
                elif self.args.edge == "max":
                    return torch.max(x[idx], dim=0)[0].reshape(1, -1)

            # e_embed = list(map(lambda i: edge_embed(i), node2edge))

            e_embed = [
                torch.sum(x[node2edge[i]], dim=0, keepdim=True)
                for i in range(len(node2edge))
            ]

            # e_embed = list(map(lambda i: sum(x[i]).reshape(1, -1), node2edge))
            # e_embed = functorch.vmap(edge_embed)(node2edge)

            # for e in torch.unique(edge_index[1]):
            #     # idx_1 = torch.where(edge_index[1] == e)[0]
            #     # idx_0 = torch.index_select(edge_index[0],-1,idx_1)
            #     # embed = torch.index_select(x,0,idx_0)
            #     # e_embed.append(sum(embed).reshape(1,-1))
            #     idx_0 = torch.index_select(edge_index[0], -1, torch.where(edge_index[1] == e)[0])
            #     e_embed.append(sum(torch.index_select(x, 0, idx_0)).reshape(1,-1))
            # edge_embed = self.edge(torch.concat((torch.stack(e_embed).squeeze(), h1), dim=1)[sample_edge_idx])
            # edge_embed = self.edge(torch.concat((torch.stack(e_embed).squeeze(), h1), dim=1))
            # edge_embed = torch.concat((torch.stack(e_embed).squeeze(), h1), dim=1)
            # edge_embed = h1
            edge_embed = self.edge(
                torch.concat((h1, torch.stack(e_embed).squeeze()), dim=1)
            )
        try:

            return x, edge_embed

        except:
            print("here")

    def get_loss(self, h1, h2, T, com_nodes):

        l1 = self.simclr_tau(h1, h2, T, com_nodes[0], com_nodes[1])
        l2 = self.simclr_tau(h2, h1, T, com_nodes[1], com_nodes[0])

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret


def glorot(tensor):
    if tensor != None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


class PMA(MessagePassing):
    """
    PMA part:
    Note that in original PMA, we need to compute the inner product of the seed and neighbor nodes.
    i.e. e_ij = a(Wh_i,Wh_j), where a should be the inner product, h_i is the seed and h_j are neightbor nodes.
    In GAT, a(x,y) = a^T[x||y]. We use the same logic.
    """

    _alpha: OptTensor

    def __init__(
        self,
        in_channels,
        hid_dim,
        out_channels,
        num_layers,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0.0,
        bias=False,
        **kwargs
    ):
        #         kwargs.setdefault('aggr', 'add')
        super(PMA, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden = hid_dim // heads
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = 0.0
        self.aggr = "add"
        #         self.input_seed = input_seed

        #         This is the encoder part. Where we use 1 layer NN (Theta*x_i in the GATConv description)
        #         Now, no seed as input. Directly learn the importance weights alpha_ij.
        #         self.lin_O = Linear(heads*self.hidden, self.hidden) # For heads combining
        # For neighbor nodes (source side, key)
        self.lin_K = Linear(in_channels, self.heads * self.hidden)
        # For neighbor nodes (source side, value)
        self.lin_V = Linear(in_channels, self.heads * self.hidden)
        self.att_r = Parameter(torch.Tensor(1, heads, self.hidden))  # Seed vector
        self.rFF = MLP(
            in_channels=self.heads * self.hidden,
            hidden_channels=self.heads * self.hidden,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=0.0,
            Normalization="None",
        )
        self.ln0 = nn.LayerNorm(self.heads * self.hidden)
        self.ln1 = nn.LayerNorm(self.heads * self.hidden)
        #         if bias and concat:
        #             self.bias = Parameter(torch.Tensor(heads * out_channels))
        #         elif bias and not concat:
        #             self.bias = Parameter(torch.Tensor(out_channels))
        #         else:

        #         Always no bias! (For now)
        self.register_parameter("bias", None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        #         glorot(self.lin_l.weight)
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.rFF.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        #         glorot(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(
        self,
        x,
        edge_index: Adj,
        size: Size = None,
        return_attention_weights=None,
        aug_weight=None,
    ):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.hidden

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in `GATConv`."
            x_K = self.lin_K(x).view(-1, H, C)
            x_V = self.lin_V(x).view(-1, H, C)
            alpha_r = (x_K * self.att_r).sum(dim=-1)
        out = self.propagate(
            edge_index, x=x_V, alpha=alpha_r, aggr=self.aggr, aug_weight=aug_weight
        )

        alpha = self._alpha
        self._alpha = None

        #         Note that in the original code of GMT paper, they do not use additional W^O to combine heads.
        #         This is because O = softmax(QK^T)V and V = V_in*W^V. So W^O can be effectively taken care by W^V!!!
        out = out + self.att_r  # This is Seed + Multihead
        # concat heads then LayerNorm. Z (rhs of Eq(7)) in GMT paper.
        out = self.ln0(out.view(-1, self.heads * self.hidden))
        # rFF and skip connection. Lhs of eq(7) in GMT paper.
        out = self.ln1(out + F.relu(self.rFF(out)))

        if isinstance(return_attention_weights, bool):
            assert alpha != None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def message(self, x_j, alpha_j, index, ptr, size_j, aug_weight):
        #         ipdb.set_trace()
        alpha = alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        if aug_weight != None:
            num_nodes = index.max() + 1
            alpha = softmax(alpha, index, ptr, num_nodes)
            alpha = alpha * aug_weight
            out_sum = scatter(alpha, index, dim=0, dim_size=num_nodes, reduce="sum")

            out_sum_index = out_sum.index_select(0, index)
            alpha = alpha / (out_sum_index + 1e-16)
        else:
            alpha = softmax(alpha, index, ptr, index.max() + 1)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def aggregate(self, inputs, index, dim_size=None, aggr=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        #         ipdb.set_trace()
        if aggr is None:
            aggr = "add"
            return scatter(inputs, index, dim=self.node_dim, reduce=aggr)
            raise ValueError("aggr was not passed!")
        return scatter(inputs, index, dim=self.node_dim, reduce=aggr)

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


class MLP(nn.Module):
    """adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout=0.5,
        Normalization="bn",
        InputNorm=False,
    ):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ["bn", "ln", "None"]
        if Normalization == "bn":
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == "ln":
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is "Identity"):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            # x = F.relu(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class HalfNLHconv(MessagePassing):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        num_layers,
        dropout,
        Normalization="bn",
        InputNorm=False,
        heads=1,
        attention=True,
    ):
        super(HalfNLHconv, self).__init__()

        self.attention = attention
        self.dropout = dropout

        if self.attention:
            self.prop = PMA(in_dim, hid_dim, out_dim, num_layers, heads=heads)
        else:
            if num_layers > 0:
                self.f_enc = MLP(
                    in_dim,
                    hid_dim,
                    hid_dim,
                    num_layers,
                    dropout,
                    Normalization,
                    InputNorm,
                )
                self.f_dec = MLP(
                    hid_dim,
                    hid_dim,
                    out_dim,
                    num_layers,
                    dropout,
                    Normalization,
                    InputNorm,
                )
            else:
                self.f_enc = nn.Identity()
                self.f_dec = nn.Identity()

    #         self.bn = nn.BatchNorm1d(dec_hid_dim)
    #         self.dropout = dropout
    #         self.Prop = S2SProp()

    def reset_parameters(self):

        if self.attention:
            self.prop.reset_parameters()
        else:
            if not (self.f_enc.__class__.__name__ == "Identity"):
                self.f_enc.reset_parameters()
            if not (self.f_dec.__class__.__name__ == "Identity"):
                self.f_dec.reset_parameters()

    #         self.bn.reset_parameters()

    def forward(self, x, edge_index, norm, aggr="add", aug_weight=None):
        # def forward(self, x, edge_index, norm, aggr='add', aug_weight=None):
        """
        input -> MLP -> Prop
        """

        if self.attention:
            x = self.prop(x, edge_index, aug_weight=aug_weight)
        else:
            x = F.relu(self.f_enc(x), inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.propagate(
                edge_index, x=x, norm=norm, aggr=aggr, aug_weight=aug_weight
            )
            # , aug_weight=aug_weight
            x = F.relu(self.f_dec(x), inplace=True)

        return x

    def message(self, x_j, norm, aug_weight):
        # return norm.view(-1, 1) * x_j
        out = x_j if aug_weight is None else x_j * aug_weight.view(-1, 1)
        return out

    def aggregate(self, inputs, index, dim_size=None, aggr=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        #       ipdb.set_trace()
        if aggr is None:
            aggr = "add"
            return scatter(inputs, index, dim=self.node_dim, reduce=aggr)
            raise ValueError("aggr was not passed!")
        # print(inputs.shape, index.shape, inputs.dtype, index.dtype)
        return scatter(inputs, index, dim=self.node_dim, reduce=aggr)
