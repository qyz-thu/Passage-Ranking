import torch
import argparse

from data_loader import DataLoader
from models import inference_model
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from bert_model import BertForSequenceEncoder
from torch.autograd import Variable
import torch.nn as nn
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


def count_correct(prob_pos, prob_neg):
    correct = 0
    prob_pos = prob_pos.view(-1).tolist()
    prob_neg = prob_neg.view(-1).tolist()
    assert len(prob_pos) == len(prob_neg)
    for step in range(len(prob_pos)):
        if prob_pos[step] > prob_neg[step]:
            correct += 1
    return correct


def eval(model, valid_reader):
    model.eval()
    correct_pred = 0
    i = 0
    for inp_tensor_qry, msk_tensor_qry, seg_tensor_qry, inp_tensor_pos, msk_tensor_pos, seg_tensor_pos, inp_tensor_neg, msk_tensor_neg, seg_tensor_neg,\
        inp_ent_pos, inp_ent_neg, pos_matrix, neg_matrix in valid_reader:
        i += 1
        print(i)
        prob_pos = model(inp_tensor_qry, msk_tensor_qry, seg_tensor_qry, inp_tensor_pos, msk_tensor_pos, seg_tensor_pos, inp_ent_pos, pos_matrix)
        prob_neg = model(inp_tensor_qry, msk_tensor_qry, seg_tensor_qry, inp_tensor_neg, msk_tensor_neg, seg_tensor_neg, inp_ent_neg, neg_matrix)
        if i % 100 == 0:
            print(i, correct_pred)
        correct_pred += count_correct(prob_pos, prob_neg)
    return correct_pred / valid_reader.total_num


def train(args, model, reader, valid_reader):
    save_path = args.outdir + '/model'
    best_acc = 0.0
    running_loss = 0.0
    t_total = int(
        reader.total_num / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    i = 0
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)
    global_step = 0
    crit = nn.MarginRankingLoss(margin=1)
    optimizer.zero_grad()
    for inp_tensor_qry, msk_tensor_qry, seg_tensor_qry, inp_tensor_pos, msk_tensor_pos, seg_tensor_pos, inp_tensor_neg, msk_tensor_neg, seg_tensor_neg,\
        inp_ent_pos, inp_ent_neg, pos_matrix, neg_matrix in reader:
        i += 1
        model.train()
        score_pos = model(inp_tensor_qry, msk_tensor_qry, seg_tensor_qry, inp_tensor_pos, msk_tensor_pos, seg_tensor_pos, inp_ent_pos, pos_matrix)
        score_neg = model(inp_tensor_qry, msk_tensor_qry, seg_tensor_qry, inp_tensor_neg, msk_tensor_neg, seg_tensor_neg, inp_ent_neg, neg_matrix)
        label = torch.ones(score_pos.size())
        if args.cuda:
            label = label.cuda()
        loss = crit(score_pos, score_neg, Variable(label, requires_grad=False))
        print(loss)
        running_loss += loss.item()
        loss.backward()
        global_step += 1
        if global_step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            logger.info('Epoch: 0, Step: %d, Loss: %.3f' % (global_step, running_loss / global_step))
            if global_step % (args.eval_step * args.gradient_accumulation_steps) == 0:
                logger.info("Start eval!")
                eval_acc = eval(model, valid_reader)
                logger.info("Dev acc: %.4f" % eval_acc)
                if eval_acc >= best_acc:
                    best_acc = eval_acc
                    torch.save({'epoch': 0,
                                'model': model.state_dict()}, save_path + ".best.pt")
                    logger.info("Saved best epoch {0}, best acc {1}".format(0, best_acc))

    print("total batch: %d" % i)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--train_path', help='train path')
    parser.add_argument('--valid_path', help='valid path')
    parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--evi_num", type=int, default=5, help='Evidence num.')
    parser.add_argument("--threshold", type=float, default=0.0, help='Evidence num.')
    parser.add_argument("--max_len", default=120, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--eval_step", default=1000, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument('--bert_pretrain', required=True)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--head_num', default=1, type=int)
    parser.add_argument('--kernal_num', default=5, type=int)
    parser.add_argument('--entity_dim', default=50, type=int)
    parser.add_argument('--GAT_hidden_dim')
    parser.add_argument('--query_len', default=10, type=int)
    parser.add_argument('--passage_len', default=60, type=int)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2"
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    handlers = [logging.FileHandler(os.path.abspath(args.outdir) + '/train_log.txt'), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logger.info('Start training!')

    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)
    logger.info("loading training set")
    reader = DataLoader(args.train_path, tokenizer, args, batch_size=args.train_batch_size)
    logger.info("loading validation set")
    valid_reader = DataLoader(args.valid_path, tokenizer, args, batch_size=args.valid_batch_size)

    bert_model = BertForSequenceEncoder.from_pretrained(args.bert_pretrain)
    bert_model = bert_model.to(device)
    model = inference_model(bert_model, args, device)
    model = model.to(device)

    train(args, model, reader, valid_reader)
