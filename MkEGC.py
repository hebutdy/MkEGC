from interactions import Interactions
from eval_metrics import *
import argparse
import logging
from time import time
import datetime
import torch
from model.MDLSJR_ATT_user import mdlsjr
import torch.nn.functional as F
import random
import pickle
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class mdlsjr(nn.Module):
    def __init__(self, user_entity, num_items, model_args, device,kg_map):
        super(mdlsjr, self).__init__()

        self.args = model_args
        self.device = device
        self.lamda = 10
        # init args
        L = self.args.L
        dims = self.args.d
        predict_T=self.args.T
        # 知识状态
        self.kg_map =kg_map
        # 用户交互状态
        self.user_embeddings = user_entity
        self.item_embeddings = nn.Embedding(num_items, dims).to(device)
        self.DP = nn.Dropout(0.5)
        #序列状态
        self.enc = DynamicGRU(input_dim=dims,
                              output_dim=dims, bidirectional=True, batch_first=True)
                              # output_dim=dims, bidirectional=False, batch_first=True)

        self.mlp = nn.Linear(dims+50*2, dims*2) # 150,100
        self.mlp2 = nn.Linear(dims+50*3, dims*2) # 150,100
        self.fc = nn.Linear(dims*2, num_items)  # 100,4551
        self.mlp_history = nn.Linear(50,50)
        self.BN = nn.BatchNorm1d(50, affine=False) #归一化为50纬度
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # BatchNorm就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布的。
    def forward(self, batch_sequences, train_len):
        #test process测试集用
        probs = []
        input = self.item_embeddings(batch_sequences)
        out_enc, h = self.enc(input, train_len)
        kg_map = self.BN(self.kg_map)
        kg_map =kg_map.detach()
        user_embeddings = self.BN(self.user_embeddings)
        user_embeddings =user_embeddings.detach()
        user = self.get_user(batch_users, user_map)
        batch_kg = self.get_kg(batch_sequences,train_len,kg_map)
        h1 = h.sum(0)
        h2 = h.squeeze()#三种状态相加
        mlp_in = torch.cat([h.sum(0), batch_kg, self.mlp_history(batch_kg), user], dim=1)
        mlp_hidden = self.mlp(mlp_in)
        mlp_hidden = torch.tanh(mlp_hidden)

        out = self.fc(mlp_hidden)
        probs.append(out)
        return torch.stack(probs, dim=1)

class DynamicGRU(BasicModule):
    def __init__(self, input_dim, output_dim,
                 num_layers=1, bidirectional=True,
                 batch_first=True):
                 # num_layers=1, bidirectional=False,
                 # batch_first=True):
        super().__init__()
        self.embed_dim = input_dim
        self.hidden_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.gru = nn.GRU(self.embed_dim,
                            self.hidden_dim,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            batch_first=self.batch_first)
        # self.attention = Attention()
        self.attention = Attention(output_dim)
        print("")

    def forward(self, inputs, lengths,one = False):
        """
        :param inputs: # (B, L, E)
        :param lengths: # (B)
        :return:
        """
        if one == True:
            hidden = lengths
            out, ht = self.gru(inputs,hidden)
        else:
            # sort data by lengths dim=0 按列排序，descending=True 从大到小排序，descending=False 从小到大排序(默认)
            _, idx_sort = torch.sort(lengths, dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            sort_embed_input = inputs.index_select(0, Variable(idx_sort))
            sort_lengths = list(lengths[idx_sort])

            inputs_packed = nn.utils.rnn.pack_padded_sequence(sort_embed_input,
                                                              sort_lengths,
                                                              batch_first=True)
            out_pack, ht = self.gru(inputs_packed)
            out = nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)
            out = out[0]
            ht = torch.transpose(ht, 0, 1)[idx_unsort]#1024,2,50
            ht = torch.transpose(ht, 0, 1)#（2,1024,50）
            out = out[idx_unsort]#（1024,49,100）

        _, att = self.attention(ht)
        ht=0.5 * ht + 0.5 * att

        return out, ht

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta
def bleu(hyps, refs):
    """
    bleu
    """
    bleu_4 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_4.append(score)
    bleu_4 = np.average(bleu_4)
    return bleu_4

def bleu_each(hyps, refs):
    """
    bleu
    """
    bleu_4 = []
    hyps=hyps.cpu().numpy()
    refs=refs.cpu().numpy()
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_4.append(score)
    return bleu_4

def precision_at_k(actual, predicted, topk,item_i):
    sum_precision = 0.0
    user = 0
    num_users = len(predicted)
    for i in range(num_users):
        if actual[i][item_i]>0:
            user +=1
            act_set = actual[i][item_i]
            pred_set = predicted[i]
            if act_set in pred_set:
                sum_precision += 1
        else:
            continue
    #print(user)
    return sum_precision / user
def Mrr(actual, predicted, item_i):
    rank = []
    user = 0
    num_users = len(predicted)
    for i in range(num_users):
        if actual[i][item_i]>0:
            user +=1
            act_set = actual[i][item_i]
            pred_set = predicted[i]
            if act_set in pred_set:
                b=pred_set.tolist()
                c=(b.index(act_set)+1)
                rank.append(c)
            else:
                rank.append(0)
        else:
            continue
    ranks=torch.tensor(rank)
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / user
    return mrr

def ndcg_k(actual, predicted, topk,item_i):
    k = min(topk, len(actual))
    idcg = idcg_k(k)
    res = 0
    user = 0
    for user_id in range(len(actual)):
        if actual[user_id][item_i] > 0:
            user +=1
            dcg_k = sum([int(predicted[user_id][j] in [actual[user_id][item_i]]) / math.log(j+2, 2) for j in range(k)])
            res += dcg_k
        else:
            continue
    #print(user)
    return res/user

def dcg_k(actual, predicted, topk):
    k = min(topk, len(actual))
    dcgs=[]
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    for user_id in range(len(actual)):
        value = []
        for i in predicted[user_id]:
            try:
                value += [topk -int(np.argwhere(actual[user_id]==i))]
                #print(value)
            except:
                value += [0]
        #dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j+2, 2) for j in range(k)])
        dcg_k = sum([value[j] / math.log(j+2, 2) for j in range(k)])
        if dcg_k==0:
           dcg_k=1e-5
        dcgs.append(dcg_k)
    return dcgs
# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

# 联合奖励
def generateReward(self, sample1, path_len, path_num, items_to_predict, pred_one_hot,h_orin,batch_kg,kg_map,tarlen):
    history_kg = self.mlp_history(batch_kg)
    Reward = []
    dist = []
    dist_replay = []
    for paths in range(path_num):
        h = h_orin
        indexes = []
        indexes.append(sample1)
        dec_inp_index = sample1
        dec_inp = self.item_embeddings(dec_inp_index)
        dec_inp = dec_inp.unsqueeze(1)
        ground_kg = self.get_kg(items_to_predict[:, self.args.T - path_len - 1:],tarlen,kg_map)
        for i in range(path_len):
            out_enc, h = self.enc(dec_inp, h, one=True)
            # out_fc = self.fc(h.squeeze())
            h1 = h.sum(0)
            h2 = h.squeeze()
            mlp_in = torch.cat([h.sum(0), batch_kg, self.mlp_history(batch_kg)], dim=1)
            # mlp_in = torch.cat([h.squeeze(), batch_kg, self.mlp_history(batch_kg)], dim=1)
            mlp_hidden = self.mlp(mlp_in)
            mlp_hidden = torch.tanh(mlp_hidden)
            out_fc = self.fc(mlp_hidden)

            out_distribution = F.softmax(out_fc, dim=1)
            out_distribution = 0.8 * out_distribution
            out_distribution = torch.add(out_distribution, pred_one_hot)
            # pai-->p(a|s)
            m = torch.distributions.categorical.Categorical(out_distribution)
            sample2 = m.sample()
            dec_inp = self.item_embeddings(sample2)
            dec_inp = dec_inp.unsqueeze(1)
            indexes.append(sample2)
        indexes = torch.stack(indexes, dim=1)
        episode_kg = self.get_kg(indexes,torch.Tensor([path_len+1]*len(indexes)),kg_map)
        '''
        dist: knowledge reward
        dist_replay: induction network training (rank)
        '''
        dist.append(self.cos(episode_kg ,ground_kg))
        dist_replay.append(self.cos(episode_kg,history_kg))
        Reward.append(bleu_each(indexes,items_to_predict[:,self.args.T-path_len-1:]))
        Reward.append(dcg_k(items_to_predict[:, self.args.T - path_len - 1:], indexes, path_len + 1))
    Reward = torch.FloatTensor(Reward).to(self.device)#（3,1024）kg奖励
    dist = torch.stack(dist, dim=0)
    dist = torch.mean(dist, dim=0)

    dist_replay = torch.stack(dist_replay, dim=0)
    dist_sort = self.compare_kgReawrd(Reward, dist_replay)#根据reward的排序，sort了dist
    Reward = torch.mean(Reward, dim=0)
    Reward = Reward + self.lamda * dist
    dist.sort =dist_sort.detach()
    return Reward, dist_sort
def train_mdlsjr(train_data, test_data, config,kg_map,user_entity):
    num_users = train_data.num_users
    num_items = train_data.num_items

    # convert to sequences, targets and users
    sequences_np = train_data.sequences.sequences #用户交互序列矩阵
    targets_np = train_data.sequences.targets
    users_np = train_data.sequences.user_ids
    trainlen_np = train_data.sequences.length
    tarlen_np = train_data.sequences.tarlen

    n_train = sequences_np.shape[0]
    logger.info("Total training records:{}".format(n_train))

    kg_map = torch.from_numpy(kg_map).type(torch.FloatTensor).to(device)
    kg_map.requires_grad=False
    user_entity = torch.from_numpy(user_entity).type(torch.FloatTensor).to(device)
    user_entity.requires_grad=False
    seq_model = mdlsjr(user_entity, num_items, config, device, kg_map).to(device)
    optimizer = torch.optim.Adam(seq_model.parameters(), lr=config.learning_rate,weight_decay=config.l2)

    lamda = 5  #loss function hyperparameter
    print("loss lamda=",lamda)
    CEloss = torch.nn.CrossEntropyLoss() # 交叉熵损失
    margin = 0.0
    MRLoss = torch.nn.MarginRankingLoss(margin=margin)#排序损失函数

    record_indexes = np.arange(n_train)
    batch_size = config.batch_size
    num_batches = int(n_train / batch_size) + 1

    stopping_step = 0
    cur_best_pre_0 = 0
    should_stop = False
    for epoch_num in range(config.n_iter):#100
        t1 = time()
        loss=0
        # set model to training mode
        seq_model.train()

        np.random.shuffle(record_indexes) #随机打乱
        epoch_reward=0.0
        epoch_loss = 0.0
        # for batchID in range(1): #1510
        for batchID in range(num_batches): #1510
            # 训练一个batch
            start = batchID * batch_size
            end = start + batch_size

            if batchID == num_batches - 1:
                if start < n_train:
                    end = n_train
                else:
                    break
            if epoch_num>=0:
                pred_one_hot = np.zeros((len(batch_users),num_items)) #1024*4551
                batch_tar=targets_np[batch_record_index]
                for i,tar in enumerate(batch_tar):
                    pred_one_hot[i][tar]=0.2/config.T   #0.2/3
                pred_one_hot = torch.from_numpy(pred_one_hot).type(torch.FloatTensor).to(device)

                prediction_score,orgin,batch_targets,Reward,dist_sort = seq_model.RLtrain(batch_sequences,
                items_to_predict,pred_one_hot,trainlen,tarlen,batch_users)

                target = torch.ones((len(prediction_score))).to(device)

                min_reward = dist_sort[0,:].unsqueeze(1)
                max_reward = dist_sort[-1,:].unsqueeze(1)
                mrloss = MRLoss(max_reward,min_reward,target)

                orgin = orgin.view(prediction_score.shape[0] * prediction_score.shape[1], -1)
                target = batch_targets.view(batch_targets.shape[0]*batch_targets.shape[1])
                reward = Reward.view(Reward.shape[0]*Reward.shape[1]).to(device)
                # torch.index_select(input, dim, index, out=None)
                prob = torch.index_select(orgin,1,target)
                prob = torch.diagonal(prob,0)
                RLloss =-torch.mean(torch.mul(reward,torch.log(prob)))
                loss = RLloss+lamda*mrloss
                epoch_loss += loss.item()

                optimizer.zero_grad()#梯度置零
                loss.backward()
                optimizer.step()
        epoch_loss /= num_batches
        t2 = time()

        if (epoch_num + 1) > 1:
            seq_model.eval()
            precision, ndcg = evaluation_mjlsjr(seq_model, train_data, test_data)
            cur_best_pre_0, stopping_step, should_stop = early_stopping(precision[0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc',
                                                                    flag_step=5)
            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            if should_stop == True:
                break
# 图卷积
class KGCN(object):
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity, adj_relation):
        self._parse_args(args, adj_entity, adj_relation)
        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.aggregator_class = SumAggregator


    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')


    def _build_model(self, n_user, n_entity, n_relation):
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.dim], initializer=KGCN.get_initializer(), name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.dim], initializer=KGCN.get_initializer(), name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.dim], initializer=KGCN.get_initializer(), name='relation_emb_matrix')

        # [batch_size, dim]
        self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)
        entities, relations = self.get_neighbors(self.item_indices)

        # [batch_size, dim]
        self.item_embeddings, self.aggregators = self.aggregate(entities, relations)

        # [batch_size]
        self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    user_embeddings=self.user_embeddings)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])
        return res, aggregators

class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data arguments
    #L: max sequence length
    #T: episode length
    parser.add_argument('--L', type=int, default=50)
    parser.add_argument('--T', type=int, default=3)

    # train arguments
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=0)

    # model dependent arguments
    parser.add_argument('--d', type=int, default=50)
    config = parser.parse_args()

    from data import Mooper
    data_set = Mooper.Mooper()

    train_set, test_set, num_users, num_items, kg_map,user_entity = data_set.generate_dataset(index_shift=1)
    train = Interactions(train_set, num_users, num_items)
    train.to_newsequence(config.L, config.T)

