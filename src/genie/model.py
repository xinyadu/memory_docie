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


        self.pair_constraints = {# ('Cognitive.IdentifyCategorize.Unspecified_Identifier', 'Contact.Contact.Correspondence_Participant'),
            # ('Cognitive.Inspection.SensoryObserve_ObservedEntity', 'Transaction.ExchangeBuySell.Unspecified_Recipient'),
            # ('Cognitive.Inspection.SensoryObserve_ObservedEntity', 'Personnel.StartPosition.Unspecified_Employee'),
            # # ('Conflict.Attack.DetonateExplode_Attacker', 'Contact.ThreatenCoerce.Unspecified_Communicator'),
            # ('Conflict.Attack.Unspecified_Attacker', 'Contact.ThreatenCoerce.Unspecified_Communicator'),
            # ('Justice.TrialHearing.Unspecified_Defendant', 'Movement.Transportation.Unspecified_Transporter'),
            # 
            ('Cognitive.IdentifyCategorize.Unspecified_Identifier', 'Contact.Contact.Correspondence_Participant'),
            # ('Conflict.Attack.DetonateExplode_Attacker', 'Contact.Contact.Meet_Participant'),
            ('Justice.Sentence.Unspecified_JudgeCourt', 'Life.Die.Unspecified_Killer'),
            # ('Conflict.Attack.DetonateExplode_Attacker', 'Contact.Contact.Unspecified_Participant')
            # ('Justice.Sentence.Unspecified_Defendant', 'Life.Injure.Unspecified_Victim')
            # ('Life.Injure.Unspecified_Instrument', 'Life.Injure.Unspecified_Instrument')
            # ('ArtifactExistence.ManufactureAssemble.Unspecified_Artifact', 'Life.Injure.Unspecified_Instrument')
            # ('ArtifactExistence.ManufactureAssemble.Unspecified_Artifact', 'ArtifactExistence.ManufactureAssemble.Unspecified_Artifact')
            # ('Cognitive.IdentifyCategorize.Unspecified_Identifier', 'Cognitive.IdentifyCategorize.Unspecified_Identifier')
            # ('ArtifactExistence.DamageDestroyDisableDismantle.Dismantle_Artifact', 'ArtifactExistence.DamageDestroyDisableDismantle.Dismantle_Artifact')
            ('Medical.Intervention.Unspecified_Patient', 'Medical.Intervention.Unspecified_Patient'), 
            ('Conflict.Attack.DetonateExplode_Attacker', 'Contact.Contact.Broadcast_Communicator'), # taliban
            ('Justice.TrialHearing.Unspecified_Defendant', 'Life.Die.Unspecified_Killer'),
            # ('Conflict.Attack.Unspecified_Attacker', 'Contact.Contact.Broadcast_Communicator') # taliban
            # ('Conflict.Attack.DetonateExplode_Attacker', 'Contact.ThreatenCoerce.Unspecified_Communicator')
            # ('Conflict.Attack.Unspecified_Attacker', 'Contact.ThreatenCoerce.Unspecified_Communicator')
            # ('Conflict.Attack.DetonateExplode_Attacker', 'Movement.Transportation.IllegalTransportation_Transporter')
            # ('Life.Die.Unspecified_Victim', 'Movement.Transportation.IllegalTransportation_Transporter')
            # ('Contact.Contact.Unspecified_Participant', 'Justice.ChargeIndict.Unspecified_Prosecutor')
            # ('Justice.ChargeIndict.Unspecified_Prosecutor', 'Justice.ChargeIndict.Unspecified_Prosecutor')
            ('Justice.ArrestJailDetain.Unspecified_Detainee', 'Movement.Transportation.Unspecified_PassengerArtifact'),
            # ('Conflict.Attack.Unspecified_Attacker', 'Contact.Contact.Unspecified_Participant')
            # ('Conflict.Attack.Unspecified_Attacker', 'Justice.ArrestJailDetain.Unspecified_Jailer'),
            # ('Justice.TrialHearing.Unspecified_Defendant', 'Movement.Transportation.Unspecified_PassengerArtifact')
            # ('Conflict.Attack.DetonateExplode_Attacker', 'Justice.ArrestJailDetain.Unspecified_Jailer')
            ('Contact.Contact.Unspecified_Participant', 'Movement.Transportation.Unspecified_PassengerArtifact'),
            ('Contact.Contact.Unspecified_Participant', 'Justice.ArrestJailDetain.Unspecified_Detainee'),
            # ('Contact.Contact.Unspecified_Participant', 'Justice.InvestigateCrime.Unspecified_Investigator') #bellingcat
            # ('Contact.RequestCommand.Unspecified_Communicator', 'Control.ImpedeInterfereWith.Unspecified_Impeder') # wiki_mass_car_bombings_0_news_9-E1
            # ('Contact.RequestCommand.Unspecified_Communicator', 'Contact.RequestCommand.Unspecified_Recipient')
            ('GenericCrime.GenericCrime.GenericCrime_Perpetrator', 'Justice.ArrestJailDetain.Unspecified_Detainee'), #wiki_drone_strikes_1_news_1-E1
            # ('GenericCrime.GenericCrime.GenericCrime_Victim', 'GenericCrime.GenericCrime.GenericCrime_Victim') #scenario_en_kairos_2-E1
            # ('GenericCrime.GenericCrime.GenericCrime_Victim', 'Life.Die.Unspecified_Victim')
        }
        self.up_constrains = {"Killer_Attacker_Injurer_Damager_Destroyer": "Killer_Attacker_Destroyer_Defendant"
        }

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
        self.up_thresh = 5
        with open("preprocessed_KAIROS/test.jsonl", 'r') as f:
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
                    for ent in ents:
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
                            if self.memory_up_cnt[doc_key][role_grp][ent][-1] >= self.up_thresh:
                                if in_id not in id_pairs_up: id_pairs_up[in_id] = []
                                id_pairs_up[in_id].append(self.memory_up_cnt[doc_key][role_grp][ent][0])
                            
        # import ipdb; ipdb.set_trace()

        # gen without id_pairs
        sample_output_no_knowledge = self.model.generate({}, {}, batch['input_token_ids'], do_sample=False, 
                                max_length=30, num_return_sequences=1,num_beams=1,)

        if self.hparams.knowledge_pair_gen:
            if self.hparams.sample_gen:
                sample_output = self.model.generate(batch['input_token_ids'], do_sample=True, 
                                    top_k=20, top_p=0.95, max_length=30, num_return_sequences=1,num_beams=1,
                                )
            else:
                sample_output = self.model.generate(id_pairs_down, id_pairs_up, batch['input_token_ids'], do_sample=False, 
                                    max_length=30, num_return_sequences=1,num_beams=1,
                                )

            # add into memory
            doc_key = batch['doc_key'][-1]
            evt_type = batch['event_type'][-1]
            pred_template = self.tokenizer.decode(sample_output.squeeze(0), skip_special_tokens=True)
            predicted_args = self.extract_args_from_template(evt_type, pred_template)
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
                            # import ipdb; ipdb.set_trace()
            
            if id_pairs_down: # if id_pairs_up:
                print(batch_idx+1)
                # print(id_pairs_down_print)
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