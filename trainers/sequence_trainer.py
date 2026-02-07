# here put the import lib
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from trainers.trainer import Trainer
from utils.utils import metric_report, metric_len_report, record_csv, metric_pop_report
from utils.utils import metric_len_5group, metric_pop_5group


class SeqTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):

        super().__init__(args, logger, writer, device, generator)
    

    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training', disable=True)
        # prog_iter = self.train_loader

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            inputs = self._prepare_train_inputs(batch)
            loss = self.model(**inputs)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



    def eval(self, epoch=0, test=False):
        if test:
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            try:
                self.model.load_state_dict(model_state_dict['state_dict'])
            except:
                self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            test_loader = self.test_loader
        else:
            desc = 'Evaluating'
            test_loader = self.valid_loader

        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)

        # ✅ 진행바 숨김
        for batch in tqdm(test_loader, desc=desc, disable=True):
            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            with torch.no_grad():
                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                pred_logits = -self.model.predict(**inputs)
                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(),
                                        aug_len=self.args.aug_seq_len, args=self.args)
        res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), self.item_pop,
                                        target_items.detach().cpu().numpy(), args=self.args)
        res_dict = {**res_dict, **res_len_dict, **res_pop_dict}

        # 텐서보드는 유지
        if not test:
            for k, v in res_dict.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar('Test/{}'.format(k), v, epoch)

        # ✅ 한 줄 요약만 로그
        # 우선순위 있는 지표만 골라서 출력(있으면)
        keys_pref = ['HR@10', 'NDCG@10', 'HR@5', 'NDCG@5']
        parts = [f'{k}={res_dict[k]:.5f}' for k in keys_pref if k in res_dict]
        if not parts:  # 없으면 숫자형 전부
            parts = [f'{k}={v:.5f}' for k, v in res_dict.items() if isinstance(v, (int, float))]
        tag = 'TEST' if test else 'EVAL'
        self.logger.info(f'[{tag}] epoch={epoch} ' + ' '.join(parts))

        if test:
            record_csv(self.args, res_dict)

        return res_dict

        


    def save_user_emb(self):

        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        try:
            self.model.load_state_dict(model_state_dict['state_dict'])
        except:
            self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        test_loader = self.test_loader

        self.model.eval()
        user_emb = torch.empty(0).to(self.device)
        desc = 'Running'

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            
            with torch.no_grad():

                per_user_emb = self.model.get_user_emb(**inputs)
                user_emb = torch.cat([user_emb, per_user_emb], dim=0)
        
        user_emb = user_emb.detach().cpu().numpy()
        import pickle
        pickle.dump(user_emb, open("./usr_emb_sasrec.pkl", "wb"))


    
    def test_group(self):

        print('')
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running Group test **********")
        desc = 'Testing'
        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        self.model.load_state_dict(model_state_dict['state_dict'])
        self.model.to(self.device)
        test_loader = self.test_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            
            with torch.no_grad():

                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                pred_logits = -self.model.predict(**inputs)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        # res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len, args=self.args)
        # res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), self.item_pop, target_items.detach().cpu().numpy(), args=self.args)
        hr_len, ndcg_len, count_len = metric_len_5group(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), [5, 10, 15, 20])
        hr_pop, ndcg_pop, count_pop = metric_pop_5group(pred_rank.detach().cpu().numpy(), self.item_pop,  target_items.detach().cpu().numpy(), [10, 30, 60, 100])

        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            self.logger.info('\t %s: %.5f' % (k, v))

        self.logger.info("User Group Performance:")
        for i, (hr, ndcg) in enumerate(zip(hr_len, ndcg_len)):
            self.logger.info('The %d Group: HR %.4f, NDCG %.4f' % (i, hr, ndcg))
        self.logger.info("Item Group Performance:")
        for i, (hr, ndcg) in enumerate(zip(hr_pop, ndcg_pop)):
            self.logger.info('The %d Group: HR %.4f, NDCG %.4f' % (i, hr, ndcg))
        
        
        return res_dict
    


class CL4SRecTrainer(SeqTrainer):

    def __init__(self, args, logger, writer, device, generator):
        
        super().__init__(args, logger, writer, device, generator)


    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            seq, pos, neg, positions, aug1, aug2 = batch
            seq, pos, neg, positions, aug1, aug2 = seq.long(), pos.long(), neg.long(), positions.long(), aug1.long(), aug2.long()
            aug = (aug1, aug2)
            loss = self.model(seq, pos, neg, positions, aug)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



class SSEPTTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):

        super().__init__(args, logger, writer, device, generator)
    

    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            seq_user, pos_user, neg_user, seq, pos, neg, positions = batch
            seq, pos, neg, positions = seq.long(), pos.long(), neg.long(), positions.long()
            seq_user, pos_user, neg_user = seq_user.long(), pos_user.long(), neg_user.long()
            loss = self.model(seq_user, pos_user, neg_user, seq, pos, neg, positions)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



    def eval(self, epoch=0, test=False):
        if test:
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            try:
                self.model.load_state_dict(model_state_dict['state_dict'])
            except:
                self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            test_loader = self.test_loader
        else:
            desc = 'Evaluating'
            test_loader = self.valid_loader

        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc, disable=True):  # ✅ 진행바 숨김
            batch = tuple(t.to(self.device) for t in batch)
            seq_user, pos_user, neg_user, seq, pos, neg, positions = batch
            seq, pos, neg, positions = seq.long(), pos.long(), neg.long(), positions.long()
            seq_user, pos_user, neg_user = seq_user.long(), pos_user.long(), neg_user.long()
            seq_len = torch.cat([seq_len, torch.sum(seq>0, dim=1)])
            with torch.no_grad():
                pred_logits = -self.model.predict(
                    seq_user, seq,
                    torch.cat([pos_user.unsqueeze(1), neg_user], dim=1),
                    torch.cat([pos.unsqueeze(1), neg], dim=1),
                    positions
                )
                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(),
                                        aug_len=self.args.aug_seq_len)
        res_dict = {**res_dict, **res_len_dict}

        if not test:
            for k, v in res_dict.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar('Test/{}'.format(k), v, epoch)

        # ✅ 한 줄 요약만 로그
        keys_pref = ['HR@10', 'NDCG@10', 'HR@5', 'NDCG@5']
        parts = [f'{k}={res_dict[k]:.5f}' for k in keys_pref if k in res_dict]
        if not parts:
            parts = [f'{k}={v:.5f}' for k, v in res_dict.items() if isinstance(v, (int, float))]
        tag = 'TEST' if test else 'EVAL'
        self.logger.info(f'[{tag}] epoch={epoch} ' + ' '.join(parts))

        if test:
            record_csv(self.args, res_dict)

        return res_dict

