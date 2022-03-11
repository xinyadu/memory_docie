import os 
import argparse 
import torch 
import logging 
import json 


import pytorch_lightning as pl 
from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from .network import BartGen
from .constrained_gen import BartConstrainedGen

from .utils import load_ontology
from collections import defaultdict 
import re

logger = logging.getLogger(__name__)

from sentence_transformers import SentenceTransformer, util
sim_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
MAX_LENGTH=512

class GenIEModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__() 
        self.hparams = args 
    

        self.config=BartConfig.from_pretrained('facebook/bart-large')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.tokenizer.add_tokens([' <arg>',' <tgr>'])

        
        if self.hparams.model=='gen':
            self.model = BartGen(self.config, self.tokenizer)
            self.model.resize_token_embeddings() 
        elif self.hparams.model == 'constrained-gen':
            self.model = BartConstrainedGen(self.config, self.tokenizer)
            self.model.resize_token_embeddings() 
        else:
            raise NotImplementedError



        self.pair_constraints = {
        ('Justice.Sentence.Unspecified_JudgeCourt', 'Life.Die.Unspecified_Victim'),
        ('Justice.Sentence.Unspecified_Defendant', 'Life.Die.Unspecified_Victim'),
        # ('Justice.TrialHearing.Unspecified_Defendant', 'Life.Injure.Unspecified_Victim'),
        # ('Contact.Contact.Broadcast_Communicator', 'Contact.ThreatenCoerce.Unspecified_Communicator'),
        ('Control.ImpedeInterfereWith.Unspecified_Impeder', 'Justice.ArrestJailDetain.Unspecified_Jailer'),
        ('Contact.RequestCommand.Unspecified_Recipient', 'Justice.ArrestJailDetain.Unspecified_Jailer'),
        ('Life.Injure.Unspecified_Injurer', 'Transaction.ExchangeBuySell.Unspecified_Giver'),
        ('Justice.TrialHearing.Unspecified_Defendant', 'Transaction.ExchangeBuySell.Unspecified_Giver'),
        ('Justice.TrialHearing.Unspecified_Defendant', 'Transaction.ExchangeBuySell.Unspecified_Recipient'),
        ## ('Conflict.Attack.DetonateExplode_Attacker', 'Contact.Contact.Meet_Participant'),
        ('Justice.Sentence.Unspecified_JudgeCourt', 'Life.Die.Unspecified_Victim'), # Justice.Sentence.Unspecified,JudgeCourt,Life.Die.Unspecified,Victim
        ('Justice.ArrestJailDetain.Unspecified_Detainee', 'Justice.ArrestJailDetain.Unspecified_Detainee'), # Justice.ArrestJailDetain.Unspecified,Detainee,Justice.ArrestJailDetain.Unspecified,Detainee
        # ('Conflict.Attack.DetonateExplode_Attacker', 'Contact.Contact.Unspecified_Participant'),
        # ('Justice.Sentence.Unspecified_Defendant', 'Life.Injure.Unspecified_Victim'), # people people (not infor.)
        # ('ArtifactExistence.ManufactureAssemble.Unspecified_Artifact', 'ArtifactExistence.ManufactureAssemble.Unspecified_Artifact'), # bomb bomb (not infor.)
        # ('ArtifactExistence.DamageDestroyDisableDismantle.Dismantle_Artifact', 'ArtifactExistence.DamageDestroyDisableDismantle.Dismantle_Artifact'), #device device (not infor.)
        # ('Life.Injure.Unspecified_Injurer', 'Life.Injure.Unspecified_Injurer'),
        # ('Justice.TrialHearing.Unspecified_Defendant', 'Justice.TrialHearing.Unspecified_Defendant'),
        ('Conflict.Attack.DetonateExplode_Attacker', 'Contact.Contact.Broadcast_Communicator'), # Conflict.Attack.DetonateExplode,Attacker,Contact.Contact.Broadcast,Communicator
        ('Conflict.Attack.Unspecified_Attacker', 'Contact.Contact.Broadcast_Communicator'), # Conflict.Attack.Unspecified,Attacker,Contact.Contact.Broadcast,Communicator
        ('Conflict.Attack.DetonateExplode_Attacker', 'Contact.ThreatenCoerce.Unspecified_Communicator'),
        ('Conflict.Attack.Unspecified_Attacker', 'Contact.ThreatenCoerce.Unspecified_Communicator'),
        # ('Conflict.Attack.DetonateExplode_Attacker', 'Movement.Transportation.IllegalTransportation_Transporter'),
        # ('Life.Die.Unspecified_Victim', 'Movement.Transportation.IllegalTransportation_Transporter'),
        # # ('Justice.ChargeIndict.Unspecified_JudgeCourt', 'Justice.TrialHearing.Unspecified_JudgeCourt'),
        # # ('Justice.ChargeIndict.Unspecified_Prosecutor', 'Justice.ChargeIndict.Unspecified_Prosecutor'),
        # ('Conflict.Attack.DetonateExplode_Attacker', 'Movement.Transportation.Unspecified_PassengerArtifact'),
        # # ('Contact.Contact.Unspecified_Participant', 'Justice.InvestigateCrime.Unspecified_Investigator'),
        # # ('Contact.RequestCommand.Unspecified_Communicator', 'Contact.RequestCommand.Unspecified_Communicator'),
        # # ('GenericCrime.GenericCrime.GenericCrime_Victim', 'GenericCrime.GenericCrime.GenericCrime_Victim'),
        # # ('GenericCrime.GenericCrime.GenericCrime_Victim', 'Life.Die.Unspecified_Victim'),
        }
        self.pair_constrains_adv = {
        ("Conflict.Attack.DetonateExplode_Attacker", "Justice.ArrestJailDetain.Unspecified_Jailer"),
        ("Life.Injure.Unspecified_Victim", "Medical.Intervention.Unspecified_Treater"),
        ("Life.Die.Unspecified_Victim", "Life.Die.Unspecified_Killer"),
        ("Life.Die.Unspecified_Victim", "Life.Injure.Unspecified_Injurer"),
        }
        if self.hparams.adv:
            self.pair_constraints = self.pair_constrains_adv
        self.up_constrains = {
        "Killer_Attacker_Injurer_Damager_Destroyer": "Killer_Attacker_Destroyer_Defendant",
        "JudgeCourt": "JudgeCourt",
        }
        self.up_thresh = 4

        self.ontology_dict = load_ontology(dataset="KAIROS")
        for key in self.ontology_dict:
            for role in self.ontology_dict[key]['arg_to_prev']:
                w = self.ontology_dict[key]['arg_to_prev'][role]
                if w == '<s>':
                    self.ontology_dict[key]['arg_to_prev'][role] = [w, 2] #</s> decoder_start_token
                else:
                    w_list = self.tokenizer.tokenize(w, add_prefix_space=True)
                    self.ontology_dict[key]['arg_to_prev'][role] = [w, self.tokenizer.encode_plus(w_list, add_special_tokens=True, add_prefix_space=True)['input_ids'][-2]]
                # if self.tokenizer
        # import ipdb; ipdb.set_trace()

        self.memory = {}
        self.memory_down = {}
        self.memory_up_cnt = defaultdict(int)
        # self.memory_place = defaultdict(int)
        
        with open("preprocessed_KAIROS0/test.jsonl", 'r') as f:
            for line in f:
                ex = json.loads(line.strip())
                doc_key = ex["doc_key"]
                evt_type = ex['event_type']
                if doc_key not in self.memory:
                    self.memory[doc_key] = {}
                    self.memory_down[doc_key] = {}
                    self.memory_up_cnt[doc_key] = {}
                # if evt_type not in self.memory[doc_key]:
                    # down
                    for evt_type in self.ontology_dict:
                        self.memory[doc_key][evt_type] = {}
                        self.memory_down[doc_key][evt_type]= {}
                        for role in self.ontology_dict[evt_type]['roles']:
                            if role not in self.memory[doc_key][evt_type]:
                                self.memory[doc_key][evt_type][role] = []
                                self.memory_down[doc_key][evt_type][role] = []
                    # up
                    for role_grp_key, role_grp in self.up_constrains.items():
                        if role_grp not in self.memory_up_cnt[doc_key]:
                            self.memory_up_cnt[doc_key][role_grp] = {} # ent1: #, ent2: #
                            
                            if role_grp_key == 'JudgeCourt':
                                ent = "George O'Toole Jr."
                                # import ipdb; ipdb.set_trace()
                                w_list = self.tokenizer.tokenize("Jr.", add_prefix_space=True)
                                out_id = self.tokenizer.encode_plus(w_list, add_special_tokens=True, add_prefix_space=True)['input_ids'][1]
                                self.memory_up_cnt[doc_key][role_grp][ent] = [out_id, self.up_thresh]

                    # place
                    # self.memory_place[doc_key] = []
        # if self.hparams.sim_train:
            # self.sim_model = SentenceTransformer('paraphrase-MiniLM-L6-v2') no other model in __init__!!
        self.all_output_templates, self.all_out_template_embs = {}, {}
        for doc_key in self.memory:
            if doc_key not in self.all_output_templates:
                self.all_output_templates[doc_key] = []
                self.all_out_template_embs[doc_key] = []


    def forward(self, inputs):
    
        return self.model(**inputs)


    def training_step(self, batch, batch_idx):
        '''
        processed_ex = {
                            'doc_key': ex['doc_key'],
                            'input_tokens_ids':input_tokens['input_ids'],
                            'input_attn_mask': input_tokens['attention_mask'],
                            'tgt_token_ids': tgt_tokens['input_ids'],
                            'tgt_attn_mask': tgt_tokens['attention_mask'],
                        }
        '''
        inputs = {
                    "input_ids": batch["input_token_ids"],
                    "attention_mask": batch["input_attn_mask"],
                    "decoder_input_ids": batch['tgt_token_ids'],
                    "decoder_attention_mask": batch["tgt_attn_mask"],   
                    "task": 0 
                }

        outputs = self.model(**inputs)
        loss = outputs[0]
        loss = torch.mean(loss)

        log = {
            'train/loss': loss, 
        } 
        return {
            'loss': loss, 
            'log': log 
        }
    

    def validation_step(self,batch, batch_idx):
        inputs = {
                    "input_ids": batch["input_token_ids"],
                    "attention_mask": batch["input_attn_mask"],
                    "decoder_input_ids": batch['tgt_token_ids'],
                    "decoder_attention_mask": batch["tgt_attn_mask"],  
                    "task" :0,   
                }
        outputs = self.model(**inputs)
        loss = outputs[0]
        loss = torch.mean(loss)

       
        
        return loss  

    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.stack(outputs))
        log = {
            'val/loss': avg_loss, 
        } 
        return {
            'loss': avg_loss, 
            'log': log 
        }
        
        
        
    def extract_args_from_template(self, evt_type, pred_template,):
        # extract argument text 
        template = self.ontology_dict[evt_type]['template']
        template_words = template.strip().split()
        predicted_words = pred_template.strip().split()
        predicted_args = defaultdict(list) # each argname may have multiple participants 
        t_ptr= 0
        p_ptr= 0 
        # evt_type = ex['event']['event_type']
        while t_ptr < len(template_words) and p_ptr < len(predicted_words):
            if re.match(r'<(arg\d+)>', template_words[t_ptr]):
                m = re.match(r'<(arg\d+)>', template_words[t_ptr])
                arg_num = m.group(1)
                try:
                    arg_name = self.ontology_dict[evt_type][arg_num]
                except KeyError:
                    print(evt_type)
                    exit() 

                if predicted_words[p_ptr] == '<arg>':
                    # missing argument
                    p_ptr +=1 
                    t_ptr +=1  
                else:
                    arg_start = p_ptr 
                    while (p_ptr < len(predicted_words)) and ((t_ptr== len(template_words)-1) or (predicted_words[p_ptr] != template_words[t_ptr+1])):
                        p_ptr+=1 
                    arg_text = predicted_words[arg_start:p_ptr]
                    predicted_args[arg_name].append(arg_text)
                    t_ptr+=1 
                    # aligned 
            else:
                t_ptr+=1 
                p_ptr+=1 
        
        return predicted_args

    def test_step(self, batch, batch_idx):

        if self.hparams.knowledge_pair_gen:
            doc_key = batch['doc_key'][-1]
            evt_type = batch['event_type'][-1]
            id_pairs_down = {}
            id_pairs_down_print = {}

            for role, ents in self.memory_down[doc_key][evt_type].items():
                in_id = self.ontology_dict[evt_type]['arg_to_prev'][role][-1]
                if ents:
                    down_out_ids = []
                    down_out_ids_print = []
                    for ent in ents[:]: # for ent in ents: limited mem
                        down_out_ids.append(ent[-1])
                        down_out_ids_print.append(ent[:-1])
                    # import ipdb; ipdb.set_trace()
                    id_pairs_down[in_id] = down_out_ids
                    id_pairs_down_print[self.ontology_dict[evt_type]['arg_to_prev'][role][0]]=down_out_ids_print

                    if role == "Participant": # fix participant exception (2 roles)
                        in_id2 = 19 # " with"
                        id_pairs_down[in_id2] = down_out_ids
                        # import ipdb; ipdb.set_trace()
            id_pairs_up = {}
            for role in self.ontology_dict[evt_type]['roles']:
                for role_grp_key, role_grp in self.up_constrains.items():
                    if role in role_grp:
                        in_id = self.ontology_dict[evt_type]['arg_to_prev'][role][-1]
                        for ent in self.memory_up_cnt[doc_key][role_grp]:
                            if self.memory_up_cnt[doc_key][role_grp][ent][-1] >= self.up_thresh and self.memory_up_cnt[doc_key][role_grp][ent][0] in batch['input_token_ids']:
                                if in_id not in id_pairs_up: id_pairs_up[in_id] = []
                                id_pairs_up[in_id].append(self.memory_up_cnt[doc_key][role_grp][ent][0])
            
            # if self.memory_place[doc_key] and "Place" in self.ontology_dict[evt_type]:
            #     role = "Place"
            #     in_id = self.ontology_dict[evt_type]['arg_to_prev'][role][-1]
            #     if in_id not in id_pairs_up: id_pairs_up[in_id] = []
            #     id_pairs_up[in_id].append(self.memory_place[doc_key][-1])

                            
        # # import ipdb; ipdb.set_trace()
        input_token_ids = batch['input_token_ids']
        if self.hparams.sim_train:
            # calculate sbert embedding and find/add most similar
            doc_key = batch['doc_key'][0]
            context_emb = sim_model.encode(batch['context_words'][0], show_progress_bar=False)
            most_sim_out_template = []
            context = batch['context_tokens'][0]
            if len(self.all_out_template_embs[doc_key])>0:
                cosine_scores = util.pytorch_cos_sim([context_emb], self.all_out_template_embs[doc_key])
                most_sim_idx = torch.argmax(cosine_scores, dim=-1)
                # if len(all_out_template_embs[doc_key])>2: import ipdb; ipdb.set_trace()
                most_sim_out_template = self.all_output_templates[doc_key][most_sim_idx]
                # most_sim_out_template = self.all_output_templates[doc_key][-1] # random ablation
                # if len(self.all_out_template_embs[doc_key])>=2: 
                    # most_sim_out_template = self.all_output_templates[doc_key][0]
            # import ipdb; ipdb.set_trace()
            # if most_sim_out_template:
            # context.extend(['</s>']+most_sim_out_template)
            context = most_sim_out_template+['</s>']+context
            # context = context
            # else:
                # context.extend(['</s>']+batch['input_template'][0])
            input_tokens = self.tokenizer.encode_plus(batch['input_template'][0], context,
                    add_special_tokens=True,
                    add_prefix_space=True,
                    max_length=MAX_LENGTH,
                    truncation='only_second',
                    padding='max_length')
            input_token_ids = torch.stack([torch.LongTensor(input_tokens['input_ids'])]) 
            if batch['input_token_ids'].device.type != 'cpu':
                input_token_ids = input_token_ids.cuda()
            # if batch_idx<5: print(input_token_ids)
            # import ipdb; ipdb.set_trace()
            # if most_sim_out_template:
            #     import ipdb; ipdb.set_trace()

        # gen without id_pairs
        sample_output_no_knowledge = self.model.generate({}, {}, input_token_ids, do_sample=False, #batch['input_token_ids']
                                max_length=30, num_return_sequences=1,num_beams=1,)

        if self.hparams.knowledge_pair_gen:
            if self.hparams.sample_gen:
                sample_output = self.model.generate(input_token_ids, do_sample=True, 
                                    top_k=20, top_p=0.95, max_length=30, num_return_sequences=1,num_beams=1,
                                )
            else:
                # id_pairs_down, id_pairs_up = {}, {}
                sample_output = self.model.generate(id_pairs_down, id_pairs_up, input_token_ids, do_sample=False, 
                                    max_length=30, num_return_sequences=1,num_beams=1,
                                )

            # add into memory
            doc_key = batch['doc_key'][-1]
            evt_type = batch['event_type'][-1]
            pred_template = self.tokenizer.decode(sample_output.squeeze(0), skip_special_tokens=True)
            predicted_args = self.extract_args_from_template(evt_type, pred_template)
            # memory_place_cache = []
            for role in predicted_args:
                for ent in predicted_args[role]:
                    if not ent: continue
                    w_list = self.tokenizer.tokenize(ent[0], add_prefix_space=True)
                    out_id = self.tokenizer.encode_plus(w_list, add_special_tokens=True, add_prefix_space=True)['input_ids'][1]
                    ent.append(out_id)
                    self.memory[doc_key][evt_type][role].append(ent)
                    # down
                    evt_type_role = "_".join([evt_type, role])
                    for pair in self.pair_constraints:
                        if evt_type_role == pair[0]:
                            evt_type2, role2 = pair[1].split("_")
                            self.memory_down[doc_key][evt_type2][role2].append(ent)
                            # if evt_type2 == 'Transaction.ExchangeBuySell.Unspecified':
                            #     import ipdb; ipdb.set_trace()
                        if evt_type_role == pair[1]:
                            evt_type2, role2 = pair[0].split("_")
                            self.memory_down[doc_key][evt_type2][role2].append(ent)
                            # if evt_type2 == 'Transaction.ExchangeBuySell.Unspecified':
                            #     import ipdb; ipdb.set_trace()
                    # up
                    for role_grp_key, role_grp in self.up_constrains.items():
                        if role in role_grp_key:
                            if ent[0] not in self.memory_up_cnt[doc_key][role_grp]:
                                self.memory_up_cnt[doc_key][role_grp][ent[0]] = [out_id, 1]
                            else:
                                self.memory_up_cnt[doc_key][role_grp][ent[0]][-1] += 1
                    # place
                    # if role == "Place":
                    #     if len(w_list)>1:
                    #         memory_place_cache.append(out_id)

            # self.memory_place[doc_key] = memory_place_cache
            
            if id_pairs_down: # if id_pairs_up:
                print(batch_idx+1)
                print(id_pairs_down_print)
                # print(id_pairs_up)
                print("ored:", self.tokenizer.decode(sample_output_no_knowledge.squeeze(0), skip_special_tokens=True))
                print("pred:", pred_template)
                print("gold:", self.tokenizer.decode(batch['tgt_token_ids'][0], skip_special_tokens=True))
            # if batch_idx == 150: import ipdb; ipdb.set_trace()
        else:
            sample_output = sample_output_no_knowledge

        sample_output = sample_output.reshape(batch['input_token_ids'].size(0), 1, -1)
        doc_key = batch['doc_key'] # list 
        tgt_token_ids = batch['tgt_token_ids']

        if self.hparams.sim_train:
            # add new output_template
            output_template = self.tokenizer.decode(sample_output[0][0], skip_special_tokens=True)
            # import ipdb; ipdb.set_trace()
            out_template_emb = sim_model.encode(output_template, show_progress_bar=False)

            space_tokenized_template = output_template.split()
            tokenized_output_template = [] 
            for w in space_tokenized_template:
                tokenized_output_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))

            self.all_output_templates[doc_key[0]].append(tokenized_output_template)
            self.all_out_template_embs[doc_key[0]].append(out_template_emb)

        return (doc_key, sample_output, tgt_token_ids) 

    def test_epoch_end(self, outputs):
        # evaluate F1 
        with open('checkpoints/{}/predictions.jsonl'.format(self.hparams.ckpt_name),'w') as writer:
            for tup in outputs:
                for idx in range(len(tup[0])):
                    
                    pred = {
                        'doc_key': tup[0][idx],
                        'predicted': self.tokenizer.decode(tup[1][idx].squeeze(0), skip_special_tokens=True),
                        'gold': self.tokenizer.decode(tup[2][idx].squeeze(0), skip_special_tokens=True) 
                    }
                    writer.write(json.dumps(pred)+'\n')

        return {} 


    def configure_optimizers(self):
        self.train_len = len(self.train_dataloader())
        if self.hparams.max_steps > 0:
            t_total = self.hparams.max_steps
            self.hparams.num_train_epochs = self.hparams.max_steps // self.train_len // self.hparams.accumulate_grad_batches + 1
        else:
            t_total = self.train_len // self.hparams.accumulate_grad_batches * self.hparams.num_train_epochs

        logger.info('{} training steps in total.. '.format(t_total)) 
        
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        # scheduler is called only once per epoch by default 
        scheduler =  get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': 'linear-schedule',
        }

        return [optimizer, ], [scheduler_dict,]