from transformers import AutoTokenizer, BertForMaskedLM
import math
import torch
from feature_class_pipe import ExtractFeatures
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from extract_surprisal_bert_pipe import get_sentences_dct
#from extract_surprisal_gpt2_pipe import filter_subtokens
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json



class Surprisal:
    def __init__(self, word_level=False, sentence_level=False):
        self.root = '/Users/matteo/Desktop/ENGKJV/text' # Change local path
        self.device =  torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = BertForMaskedLM.from_pretrained("bert-base-cased").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.par_data, self.sent_data = self.create_bible_dict()
        self.ctx = 512
        self.word_level = word_level
        self.sentence_level = sentence_level
        print(f"--Using device: {self.device}")


    def create_bible_dict(self):
        """
        Creates a dictionary for the ENGKJV Bible collection
        from normalized .txt files. Returns two dictionaries with 
        chapter numbers as keys and Bible paragraphs as values, 
        both as a string of text (paragraph), and as and list 
        of strings (sentences).
        """
        par_dct = {}
        sent_dct = {}
        filenames = sorted(os.listdir(self.root))
        for i, file in enumerate(filenames):
            filepath = os.path.join(self.root, file)
            txt = self.open_txt_files(filepath)
            paragraph_sent = txt.split('\n')
            paragraph_txt = ' '.join(paragraph_sent)
            par_dct[i+1] = paragraph_txt
            sent_dct[i+1] = paragraph_sent
        return par_dct, sent_dct


    def open_txt_files(self, filepath):
        with open(filepath) as f:
            return f.read()
        
    # NEW
    def reconstruct_with_indices(self, input_ids, debug=False):
        rec_indices = []                # final list of dicts
        buffer = []                     # holds sub‑tokens that start with "##"
        start_idx = None                # index of the first sub‑token in the buffer

        for i in range(len(input_ids)):
            tok = self.tokenizer.convert_ids_to_tokens(input_ids[i].item())
            if tok.startswith("##"):
                if start_idx is None:
                    start_idx = i - 1  
                buffer.append(tok)      
            else:
               
                if buffer:
                    first_piece = rec_indices[-1]['token']
                    if isinstance(first_piece, list):
                        combined = first_piece + buffer
                    else:
                        combined = [first_piece] + buffer
                    rec_indices[-1] = {'idx': start_idx, 'token': combined}
                    buffer = []
                    start_idx = None
                rec_indices.append({'idx': i, 'token': tok})

        if buffer:
            start_idx = start_idx if start_idx is not None else rec_indices[-1]['idx']
            first_piece = rec_indices[-1]['token']
            combined = ([first_piece] if not isinstance(first_piece, list) else first_piece) + buffer
            rec_indices[-1] = {'idx': start_idx, 'token': combined}

        if debug:
            contract = self.merge_contractions(rec_indices)
            print(f'\n==== Full Merged ====')
            for k in enumerate(contract):
                print (k)
            print(f'=======================')
            m = self._convert_merged_tok_to_ids(contract)
            for k, v in m.items():
                print((k, self.convert_ids_to_tok(v)))

        merge_toks = self.merge_contractions(rec_indices)
        return self._convert_merged_tok_to_ids(merge_toks)

    # NEW
    def _convert_merged_tok_to_ids(self, rec_indices):
        result ={}
        for entry in rec_indices:
            if isinstance(entry['token'], list):
                tok = [self.tokenizer.convert_tokens_to_ids(t) for t in entry['token']]
                result[entry['idx']] = tok
            else:
                tok = self.tokenizer.convert_tokens_to_ids(entry['token'])
                result[entry['idx']] = tok
        return result

    # NEW
    @staticmethod
    def flatten_ids(ids):
        token_ids = [t for w in ids for t in (w if isinstance(w, list) else [w])]
        return token_ids
    
    # NEW
    def convert_ids_to_tok(self, t):
        if isinstance(t, list):
            return [self.tokenizer.convert_ids_to_tokens(x) for x in t]
        return [self.tokenizer.convert_ids_to_tokens(t)]
    
    # NEW
    def create_word_level_window(self, input, chap, inputs_ids, i, left, right, debug=True):
        merge_tokens = self.reconstruct_with_indices(inputs_ids)
        word_pos = sorted([pos for pos, ids in merge_tokens.items() if ids not in (
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id
            )])
       
        # CHECK MERGED WORDS
        stats = self.get_word_statistics(input, print_on_screen=False)
        chap_stats = [dct for dct in stats if dct['chap_id'] == chap]
        for sen in chap_stats[0].get('txt'):
            expected_length = sen
        actual_length = [self.convert_ids_to_tok(merge_tokens[k]) for k in word_pos]
        assert len(actual_length) == expected_length, f'Number of merged words mismatch with input words.'
        #print(f'=== Word Level Check ===')
        #print(f'Chapter: {chap}')
        #print(f'Expected word count: {expected_length}')
        #print(f'Actual word_pos count: {len(actual_length)}')
        #print(f'Match: {expected_length == len(actual_length)}')
        #print(f'========================')

        if i in word_pos:
            j = word_pos.index(i)
            if j >= left and len(word_pos) - j - 1 >= right:
          
                token_indices = word_pos

                # full contexts
                left_ctx_words = [merge_tokens[k] for k in token_indices if k < i]
                right_ctx_words = [merge_tokens[k] for k in token_indices if k > i]

                target_word = merge_tokens[i]

                # VISIBLE IDS UNDER INFERENCE
                visible_left_words = left_ctx_words[-left:]
                visible_right_words = right_ctx_words[:right]

                # FLATTEN IDS TO LIST
                full_left_tokens = self.flatten_ids(left_ctx_words)
                full_right_tokens = self.flatten_ids(right_ctx_words)

                visible_left_tokens = self.flatten_ids(visible_left_words)
                visible_right_tokens = self.flatten_ids(visible_right_words)

                target_tokens = target_word if isinstance(target_word, list) else [target_word]

                # MASKING
                mask_left = len(full_left_tokens) - len(visible_left_tokens)
                mask_right = len(full_right_tokens) - len(visible_right_tokens)

                window = (
                    [self.tokenizer.cls_token_id]
                    + [self.tokenizer.mask_token_id] * mask_left
                    + visible_left_tokens
                    + [self.tokenizer.mask_token_id] * len(target_tokens)
                    + visible_right_tokens
                    + [self.tokenizer.mask_token_id] * mask_right
                    + [self.tokenizer.sep_token_id]
                )
                assert len(window) == len(inputs_ids), f"Context window and input length do not match: {len(window)} | {len(inputs_ids)}"

                if debug:
                    print("\n=== DEBUG INFO ===")
                    print(f'[1] CONTEXTS:')
                    print(f'    Tokens Left Context   | {left}')
                    print(f'    Tokens Right Context  | {right}')
                    print(f'[2] FULL CONTEXTS:')
                    print(f'    Full Left Context     | {self.convert_ids_to_tok(left_ctx_words)}')
                    print(f'    Full Right Context    | {self.convert_ids_to_tok(right_ctx_words)}')
                    print(f'[3] MASKED TOKEN:')
                    print(f'    Masked IDs            | {target_tokens}')
                    print(f'    Masked Token          | {self.convert_ids_to_tok(target_tokens)}')
                    print(f'[4] VISIBLE TOKENS UNDER INFERENCE')
                    print(f'    Visible Left IDs      | {visible_left_tokens}')
                    print(f'    Visible Left          | {self.convert_ids_to_tok(visible_left_tokens)}')
                    print(f'    Visible Right         | {self.convert_ids_to_tok(visible_right_tokens)}')
                    print(f'[5] MASKED WINDOW')
                    print(f'    Masked IDs            | {window}')
                    print(f'    Masked Window:        | {self.convert_ids_to_tok(window)}')
                    print(f'    Not Masked            | {[self.convert_ids_to_tok(merge_tokens[id]) for id in token_indices]}')
                    print(f'    Masked Length         | {len(window)}')
                    print(f'    Not Masked Length     | {len(self.flatten_ids([merge_tokens[id] for id in token_indices]))}')
                    print("===================\n")

                new_input = torch.tensor(window).unsqueeze(0).to(self.device)
                att = torch.ones_like(new_input).to(self.device)
                masked_input = {'input_ids': new_input, 'attention_mask': att}
                outputs = self.model(**masked_input)
                logits = outputs['logits']

                mask_pos = 1 + mask_left + len(visible_left_tokens)  
                
                assert masked_input['input_ids'][0, mask_pos] == self.tokenizer.mask_token_id, f"Masked IDs are not at the expected position {mask_pos}." 
                
                # WORD LEVEL INFERENCE
                bert_tokens = []
                for sub_pos, ids in enumerate(target_tokens):
                    if debug:
                        print(f'Subtokens: {sub_pos} | IDs: {ids}')
                    pos = mask_pos + sub_pos
                    probs = torch.softmax(logits[0, pos], dim=-1)
                    token_prob = probs[ids].item()
                    surprisal = -math.log2(token_prob)
                    subtoken = self.tokenizer.convert_tokens_to_string(self.convert_ids_to_tok(ids))
                    bert_tokens.append({
                        'form': subtoken,
                        'surprisal': surprisal
                    })
                
                assert len(target_tokens) == len(bert_tokens), f"Target tokens number mismatch."

                word_dct = {
                    'form': self.tokenizer.convert_tokens_to_string(self.convert_ids_to_tok(target_tokens)),
                    'surprisal': sum([s['surprisal'] for s in bert_tokens]),
                    'bert_tokens': bert_tokens
                    }

                if debug:
                    print(f'==== DEBUG INFO =====')
                    print(f'Masked Position:    {mask_pos}      | {merge_tokens[mask_pos]} | {target_tokens}')
                    print(f'Target Token ID:    {target_tokens} | {self.convert_ids_to_tok(target_tokens)}')
                    print(f'Subtoken Surprisal: {bert_tokens}')
                    print(f'=====================')

                return word_dct
            
    # NEW
    def estimate_word_surprisal(self, input):
        self.model.eval()
        wordpiece_surp = []

        print('--Evaluating word-level surprisal ...')
        with torch.no_grad():
            for chap, txt in tqdm(input.items(), desc='Processing', unit='word'):

                chap_dct = {'chap_id': chap,
                            'sentences': []
                            }
                
                inputs = self.tokenizer(txt, return_tensors='pt', 
                                        max_length=self.ctx, 
                                        truncation=True, 
                                        padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()} 
                inputs_ids = inputs['input_ids'][0]

                context_windows = [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]    # Extend context as needed                
                for left, right in context_windows:

                    context_dct = {'context_window': str(left)+'L|'+str(right)+'R',
                                   'words': []
                                   }
                    
                    start = left + 1 
                    end = len(inputs_ids) - right - 1 
                    for i in range(start, end):
                        word_surp = self.create_word_level_window(input, chap, inputs_ids, i, left, right, debug=False)
                        if word_surp:
                            context_dct['words'].append(word_surp)
                    chap_dct['sentences'].append(context_dct)
                wordpiece_surp.append(chap_dct)
        return wordpiece_surp

    # NEW
    def create_sentence_level_window(self, curr_sent_ids, sent_idx, chap, input, i, right_idx, debug=True):
        merge_tokens = self.reconstruct_with_indices(curr_sent_ids)
        word_pos = sorted([pos for pos, ids in merge_tokens.items() if ids not in (
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id
            )])
        
        # CHECK MERGED WORDS
        stats = self.get_word_statistics(input, print_on_screen=False)
        chap_stats = [dct for dct in stats if dct['chap_id'] == chap]
        for sen in chap_stats[0].get('sentences'):
            if sen.get('sentence_id') == sent_idx:
                expected_length = sen.get('word_count')
        actual_length = [self.convert_ids_to_tok(merge_tokens[k]) for k in word_pos]
        assert len(actual_length) == expected_length, f'Number of merged words mismatch with input words.'
        #print(f'=== Sentence Level Check ===')
        #print(f'Chapter: {chap} | Sentence: {sent_idx}')
        #print(f'Expected word count: {expected_length}')
        #print(f'Actual word_pos count: {len(actual_length)}')
        #print(f'Match: {expected_length == len(actual_length)}')
        #print(f'============================')
    
        if i in word_pos:
            token_indices = word_pos

            # FULL CONTEXTS
            left_ctx_words = [merge_tokens[k] for k in token_indices if k < i]
            right_ctx_words = [merge_tokens[k] for k in token_indices if k > i]

            target_word = merge_tokens[i]

            # VISIBLE IDS UNDER INFERENCE
            visible_left_words = left_ctx_words[:i]
            if right_idx == 'S':
                visible_right_words = right_ctx_words
            else:
                visible_right_words = right_ctx_words[:right_idx]

            # FLATTEN IDS TO LIST
            full_left_tokens = self.flatten_ids(left_ctx_words)
            full_right_tokens = self.flatten_ids(right_ctx_words)

            visible_left_tokens = self.flatten_ids(visible_left_words)
            visible_right_tokens = self.flatten_ids(visible_right_words)

            target_tokens = target_word if isinstance(target_word, list) else [target_word]

            # MASKING
            mask_left = len(full_left_tokens) - len(visible_left_tokens)
            mask_right = len(full_right_tokens) - len(visible_right_tokens)

            window = (
                [self.tokenizer.cls_token_id]
                + [self.tokenizer.mask_token_id] * mask_left
                + visible_left_tokens
                + [self.tokenizer.mask_token_id] * len(target_tokens)                   
                + visible_right_tokens
                + [self.tokenizer.mask_token_id] * mask_right
                + [self.tokenizer.sep_token_id]
            )
            if debug:
                print("\n=== DEBUG INFO ===")
                print(f'[1] CONTEXTS:')
                print(f'    Tokens Left Context   | {left_ctx_words}')
                print(f'    Tokens Right Context  | {right_idx}')
                print(f'[2] FULL CONTEXTS:')
                print(f'    Full Left Context     | {self.convert_ids_to_tok(left_ctx_words)}')
                print(f'    Full Right Context    | {self.convert_ids_to_tok(right_ctx_words)}')
                print(f'[3] MASKED TOKEN:')
                print(f'    Masked IDs            | {target_tokens}')
                print(f'    Masked Token          | {self.convert_ids_to_tok(target_tokens)}')
                print(f'[4] VISIBLE TOKENS UNDER INFERENCE')
                print(f'    Visible Left IDs      | {visible_left_tokens}')
                print(f'    Visible Left          | {self.convert_ids_to_tok(visible_left_tokens)}')
                print(f'    Visible Right         | {self.convert_ids_to_tok(visible_right_tokens)}')
                print(f'[5] MASKED WINDOW')
                print(f'    Masked IDs            | {window}')
                print(f'    Masked Window:        | {self.convert_ids_to_tok(window)}')
                print(f'    Not Masked            | {[self.convert_ids_to_tok(merge_tokens[id]) for id in token_indices]}')
                print(f'    Masked Length         | {len(window)}')
                print(f'    Not Masked Length     | {len(self.flatten_ids([merge_tokens[id] for id in token_indices]))}')
                print("===================\n")

            new_input = torch.tensor(window).unsqueeze(0).to(self.device)
            att = torch.ones_like(new_input).to(self.device)
            masked_input = {'input_ids': new_input, 'attention_mask': att}
            outputs = self.model(**masked_input)
            logits = outputs['logits']

            mask_pos = 1 + mask_left + len(visible_left_tokens)  
            
            assert masked_input['input_ids'][0, mask_pos] == self.tokenizer.mask_token_id, f"Masked IDs are not at the expected position {mask_pos}." 

            # SENTENCE LEVEL INFERENCE
            bert_tokens = []
            for sub_pos, ids in enumerate(target_tokens):
                if debug:
                    print(f'Subtokens: {sub_pos} | IDs: {ids}')
                pos = mask_pos + sub_pos
                probs = torch.softmax(logits[0, pos], dim=-1)
                token_prob = probs[ids].item()
                surprisal = -math.log2(token_prob)
                subtoken = self.tokenizer.convert_tokens_to_string(self.convert_ids_to_tok(ids))
                bert_tokens.append({                        
                    'form': subtoken,
                    'surprisal': surprisal
                    })
                
            assert len(target_tokens) == len(bert_tokens), f"Target tokens number mismatch."

            word_dct = {
                'form': self.tokenizer.convert_tokens_to_string(self.convert_ids_to_tok(target_tokens)),
                'surprisal': sum([s['surprisal'] for s in bert_tokens]),
                'bert_tokens': bert_tokens
                }

            if debug:
                print(f'==== INFERENCE INFO =====')
                print(f'Masked POS: {mask_pos}      | {merge_tokens[i]} | {target_tokens}')
                print(f'Target ID:  {target_tokens} | {self.convert_ids_to_tok(target_tokens)}')
                print(f'Surprisal:  {bert_tokens}')
                print(f'=====================')

            return word_dct

    # UPDATED
    def estimate_sentence_surprisal(self, input):
        """
        Estimates surprisal at the word (subtoken) level.
        It takes a list of sentences (variable 'txt') and estimates 
        surprisal incrementally with varying size of the right context
        (i.e., 0 word, 1 word, 2 word, and all words 'S').
        """
        self.model.eval()
        wordpiece_surp = []    # UPDATED

        print('--Evaluating sentence-level surprisal...')
        with torch.no_grad():
            for chap, txt in tqdm(input.items(), desc='Processing', unit='chapter'):
                inputs = self.tokenizer(txt, return_tensors='pt', 
                                        max_length=self.ctx, 
                                        truncation=True, 
                                        padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()} 

                chap_entry = {"chap_id": chap,  # UPDATED
                              "sentences": []
                              }

                tokenized = []
                for ids, att in zip(inputs["input_ids"], inputs["attention_mask"]):
                    length = att.sum().item()                                   # Skip padding tokens 
                    clean_ids = ids[1:length-1]                                 # Skip CLS + SEP tokens in each sentence
                    tokenized.append(clean_ids)

                for sentence in range(len(tokenized)):                          # loop over sentences
                    curr_sent_ids = tokenized[sentence]

                    sentence_entry = {'sentence_id': sentence+1,
                                      'contexts': []
                                      }
                    
                    for right_idx in (0, 1, 2, "S"):
                        # This prevents the right context to exceed the sentence lenght
                        # So always 0/1/2 words on the right of the MASK tokens
                        # For idx=0, the period '.' is the last MASK token
                        # For idx=1, the token before the period is the last MASK toke
                        # For idx=2, the third to the last token is the MASK token
                        context_entry = {
                            "context_window": 'SL|'+str(right_idx)+'R',
                            "words": []
                            }

                        for idx in range(len(curr_sent_ids)):
                            total_right = len(curr_sent_ids) - idx -1 
                            if right_idx != "S" and total_right < right_idx:
                                continue
                            sent_surp = self.create_sentence_level_window(curr_sent_ids, sentence, chap, input, idx, right_idx, debug=False)
                            if sent_surp:
                                context_entry['words'].append(sent_surp)
                        sentence_entry['contexts'].append(context_entry)
                    chap_entry['sentences'].append(sentence_entry)
                wordpiece_surp.append(chap_entry)
        return wordpiece_surp

    # NEW
    def merge_contractions(self, words):
        suffixes = ('m', 's', 't', 'd', 're', 've', 'll')
        merged = []
        i = 0
        while i < len(words):
            if i + 2 < len(words):
                first, second, third = words[i], words[i+1], words[i+2]

                # Merge plural form of multiple subtokens (i.e., ["bi", "##scu", "##its"] + "'")
                if isinstance(first.get('token'), list) and second.get('token') == "'":
                    merged.append({
                       "idx": first["idx"],
                        "token": first["token"] + [second.get('token')]
                    })
                    i += 2                                            
                    continue

                # Merge plural form of single subtoken when the third token is a list (i.e., "Gods" + "'")
                if second.get('token') == "'" and isinstance(third.get('token'), list):
                    merged.append({
                       "idx": first["idx"],
                        "token": [first["token"], second.get('token')]
                    })
                    i += 2                    
                    continue

                first_tok = first.get('token')
                second_tok = second.get('token')
                third_tok  = third.get('token')

                # Merge single tokens + "'" + suffixes (i.e., "God ' s", "We ' d", "I ' ll")
                if second_tok == "'" and third_tok in suffixes:
                    merged.append({"idx": first["idx"], 
                                   "token": [first_tok, second_tok, third_tok]
                                   })
                    i += 3
                    continue

                # Merge plural forms of single tokens when the third token is a single subtoken
                if second_tok == "'" and third_tok not in suffixes:
                    merged.append({"idx": first["idx"], 
                                   "token": [first_tok, second_tok]
                                   })      
                    i += 2
                    continue 

            merged.append(words[i])
            i += 1
        return merged

    # NEW
    def get_word_statistics(self, sent, print_on_screen=True):
        """
        Check merged words during inference.
        """
        stats_sent = []
        total_wc = 0

        for chapter, sent_lst  in sent.items():
            chap_entry = {'chap_id': chapter, 'txt': [], 'sentences': []}

            sent_wc = 0

            for i in range(len(sent_lst)):
                tokenized_period = []
                sent = sent_lst[i].split()            
                for word in sent:
                    if word.endswith("."):
                        tokenized_period.append(word[:-1])
                        tokenized_period.append(word[-1])
                    else:
                        tokenized_period.append(word)

                sent_entry = {
                    'sentence_id': i,
                    'n_sentences': len(sent_lst),
                    'tokenized_sent': tokenized_period,
                    'word_count': len(tokenized_period), 
                    'original_word_count': len(sent)
                    }
                
                sent_wc += len(tokenized_period)   
                chap_entry['txt'].append(len(tokenized_period))
                chap_entry['sentences'].append(sent_entry)
            total_wc += sent_wc
            stats_sent.append(chap_entry)

        if print_on_screen:
            print('\n==== DEBUG INFO ====')
            print(f'Total Bible Collection of {len(stats_sent)} Chapters | Total Word Count: {total_wc}')      
            for dct in stats_sent:
                chap_id = dct.get('chap_id')
                txt_length = sum(dct.get('txt'))
                print(f'Total Words in chapter ID {chap_id}: {txt_length}')
                for sent_data in dct['sentences']: 
                    print(f'Chapter ID: {chap_id} | Total Sentences: {sent_data.get("n_sentences")} || Sentence ID: {sent_data.get("sentence_id")} | Words: {sent_data.get("word_count")}')
                    print(f'Tokenized: {sent_data.get('tokenized_sent')}')     
            print('====================')
        return stats_sent
   
# NEW
def reconstruct_word_surp_dct(word_surp):
    result = []
    for entry in word_surp:
        #c = {}
        chap_id = entry['chap_id']
        for sentence in entry['sentences']:
            context_window = sentence['context_window']
            for word_array in sentence['words']:
                if word_array:
                    word_obj = word_array
                    result.append({
                        'word': word_obj['form'],
                        'surprisal': word_obj['surprisal'],
                        'chap': chap_id,
                        'context_window': context_window  
                })
    return result

# NEW
def reconstruct_sent_surp_dct(word_surp):
    result = []
    for entry in word_surp:
        chap_id = entry['chap_id']
        for sentence in entry['sentences']:
            sent_id = sentence['sentence_id']
            contexts = sentence['contexts']
            for cw_dct in contexts:
                cw = cw_dct['context_window']
                for word_array in cw_dct['words']:
                    word_obj = word_array
                    result.append({
                    'word': word_obj['form'],
                    'surprisal': word_obj['surprisal'],
                    'chap': chap_id,
                    'context_window': cw,
                    'sent_id': sent_id
                })
    return result


def main(test=True, write_to_csv=False, write_to_json=False):
    model = Surprisal(word_level=True, sentence_level=True)
    par = model.par_data                  # Full 260 chapters texts
    sent = model.sent_data                # Full 260 chapters tokenized by sentences
    
    # Test shorter texts
    if test:
        #par = {260: par.get(260)}        # Choose a number between 1-260    
        #sent = {260: sent.get(260)}

        par = {260: ["God is an absolute misunderstood concept indeed.\
                     Well I don't remember the circumstances.\
                     What on earth are we doing here.\
                     We'd retain knowledge of Gods' unbearable pain and dogs' love and biscuits' flavours but I'll consider that.\
                     I love God's gift.\
                     I love unbelievable God's gift.\
                     I endulge on unbelievable biscuits' flavours."],
                     200: ["Invocation and ritual dance of the young pumpkin. You're probably wondering why I'm here."]
        }
        sent = {260: ["God is an absolute misunderstood concept indeed.",
                      "Well I don't remember the circumstances.",
                      "What on earth are we doing here.",
                      "We'd retain knowledge of Gods' unbearable pain and dogs' love and biscuits' flavour but I'll consider that.",
                      "I love God's gift.",
                      "I love unbelievable God's gift.",
                      "I endulge on unbelievable biscuits' flavours."],
                    200: ["Invocation and ritual dance of the young pumpkin.", "You're probably wondering why I'm here."]
        }

    if model.word_level:
        surp_word = model.estimate_word_surprisal(par)   
    if model.sentence_level:
        surp_sent = model.estimate_sentence_surprisal(sent)

    if write_to_csv:
        data_word = reconstruct_word_surp_dct(surp_word)
        data_sent = reconstruct_sent_surp_dct(surp_sent)
        df_word= pd.DataFrame(data=data_word)
        df_sent = pd.DataFrame(data=data_sent)
        df_word.to_csv('word_level_merged_full_chap_260_csv.csv', sep='\t')
        df_sent.to_csv('sent_level_merged_full_chap_260_csv.csv', sep='\t')

    if write_to_json:
        word_level_surprisal_js = json.dumps(surp_word, indent=3)
        sentence_level_surprisal_js = json.dumps(surp_sent, indent=3)
        
        with open("word_level_surp.json", "w") as f:
            f.write(word_level_surprisal_js)

        with open("sentence_level_surp.json", "w") as f:
            f.write(sentence_level_surprisal_js)

    return surp_word, surp_sent


if __name__ == '__main__':  
    surp_word, surp_sent = main(test=True, write_to_csv=True, write_to_json=True) 
    