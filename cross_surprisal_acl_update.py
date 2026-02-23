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
        

    def create_surprisal_window(self, inputs_ids, i, left, right):
        """ 
        Creates a context window at the word (subtokens) level with 
        format '[CLS] left [MASK] right [SEP]', and estimates the surprisal 
        value of the masked token. It takes a tensor of inputs ids, the current 
        position [i], and the left and right indices. Returns a tuple with 1) the 
        WordPiece subtoken, and 2) its surprisal value.
        """
        window = [self.tokenizer.cls_token_id]      # Initialize empty window with [CLS]
        for k in range(left, 0, -1):                # Left context
            window.append(inputs_ids[i-k].item())

        window.append(self.tokenizer.mask_token_id)

        for k in range(1, right+1):                 # Right context
            window.append(inputs_ids[i+k].item())

        window.append(self.tokenizer.sep_token_id)
        
        #DEBUG
        #print(f'-- Window: {window}')

        new_input = torch.tensor(window).unsqueeze(0).to(self.device)
        att = torch.ones_like(new_input).to(self.device)
        masked_input = {'input_ids': new_input, 'attention_mask': att}
        outputs = self.model(**masked_input)
        logits = outputs['logits']
        mask_pos = 1 + left                                             # CHANGE Masked position is always i+1
        probs = torch.softmax(logits[0, mask_pos], dim=-1)

        token_id = inputs_ids[i].item()                                 # CHANGE added .item()
        token_prob = probs[token_id].item()
        surprisal = -math.log2(token_prob)

        token = self.tokenizer.convert_ids_to_tokens(token_id)

        # DEBUG
        print("-"*50)
        print(f"--Context Window: {left}wL/{right}R | Masked Token: [{token}]")
        print(self.tokenizer.convert_ids_to_tokens(new_input[0].tolist()))
        return token, surprisal


    def create_sentence_window(self, curr_sent_ids, sentence, idx, right_idx, right_ctx):
        """
        Create a context window at the sentence level with asymmetrical 
        right context: [CLS] full_sentence [MASK] 0_word / 1_word / 2/word / full_sentence [SEP]
        where the [MASK] token is always the 1st word of Ssentence Si+1 (Note that the 
        function's logic here need to be optimized). Returns a dictionary with the surprisal scores
        for each masked token.
        """
        left_ctx = curr_sent_ids[:idx]
    
        window = [self.tokenizer.cls_token_id]
        window += left_ctx
        window.append(self.tokenizer.mask_token_id)
        window += right_ctx                             
        window.append(self.tokenizer.sep_token_id)

        new_input = torch.tensor(window).unsqueeze(0).to(self.device)
        att = torch.ones_like(new_input).to(self.device)
        masked_input = {'input_ids': new_input, 'attention_mask': att}
        outputs = self.model(**masked_input)
        logits = outputs['logits']

        #print('--Att shape:')
        #print(att.shape)
        #print('--New Input shape:')
        #print(new_input.shape)

        mask_pos = 1 + len(left_ctx)
        probs = torch.softmax(logits[0, mask_pos], dim=-1)

        target = curr_sent_ids[idx].item()
        print('--Current Mask:', self.tokenizer.convert_ids_to_tokens(target))
        token_prob = probs[target].item()
        surprisal = -math.log2(token_prob)

        token = self.tokenizer.convert_ids_to_tokens(target)
        # DEBUG
        print("-"*50)
        print(f"--Current Context Window for sentence {sentence}: L:1S/R:{right_idx}W | Masked Token: [{token}]")
        print(self.tokenizer.convert_ids_to_tokens(new_input[0].tolist()))
        return token, surprisal


    def estimate_surprisal(self, input):
        """
        Estimates surprisal values at the word (subtoken) level
        based on the current context window used. It takes a 
        dictionary as input, where the variable 'txt' is the 
        Bible paragraph represented as a string of text. 
        """
        self.model.eval()
        wordpiece_surp = defaultdict(list)

        print('--Evaluate surprisal...')
        with torch.no_grad():
            for chap, txt in tqdm(input.items(), desc='Processing', unit='chapter'):
                inputs = self.tokenizer(txt, return_tensors='pt', 
                                        max_length=self.ctx, 
                                        truncation=True, 
                                        padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()} 

                #outputs = self.model(**inputs) # extract the logits    # CHANGE from previos suprisal function
                #logits = outputs['logits']                             # CHANGE from previous surprisal function 
                inputs_ids = inputs['input_ids'][0]
                #att = inputs['attention_mask'][0]                      # CHANGE from previous surprisal function

                # WORD-LEVEL SURPISAL
                context_windows = [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]    # Extend context as needed
                for left, right in context_windows:
                    start = left + 1
                    end = len(inputs_ids) - right - 1 
                    for i in range(start, end):
                        token, surp = self.create_surprisal_window(inputs_ids, i, left, right)
                        wordpiece_surp[str(left)+'L/'+str(right)+'R'].append((token, surp))

        return wordpiece_surp


    def estimate_sentence_surprisal(self, input):
        """
        Estimates surprisal at the word (subtoken) level.
        It takes a list of sentences (variable 'txt') and estimates 
        surprisal incrementally with varying size of the right context
        (i.e., 0 word, 1 word, 2 word, and all words 'S').
        """
        self.model.eval()
        wordpiece_surp = defaultdict(list)

        print('--Evaluate surprisal...')
        with torch.no_grad():
            for chap, txt in tqdm(input.items(), desc='Processing', unit='chapter'):
                inputs = self.tokenizer(txt, return_tensors='pt', 
                                        max_length=self.ctx, 
                                        truncation=True, 
                                        padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()} 

                tokenized = []
                for ids, att in zip(inputs["input_ids"], inputs["attention_mask"]):
                    length = att.sum().item()           # Skip padding tokens 
                    clean_ids = ids[1:length-1]         # Skip CLS + SEP tokens in each sentence
                    tokenized.append(clean_ids)

                    #DEBUG
                    #print('--Input:', tokenized)

                for sentence in range(len(tokenized)):               # loop over sentences
                    curr_sent_ids = tokenized[sentence]
                    print('--Current sentence:')
                    print(self.tokenizer.convert_ids_to_tokens(tokenized[sentence]))

                    for idx in range(len(curr_sent_ids)):
                        for right_idx in [0, 1, 2, "S"]:                        # 0 word, 1 word, 2 word for right context and assymetrical context "S"
                            if right_idx == "S":
                                right_ctx = curr_sent_ids[idx+1:]               # Comprises all the r
                            else:
                            #if len(curr_sent_ids) <= right:
                            #    continue
                                right_ctx = curr_sent_ids[idx+1:idx+1+right_idx]
                        
                            token, surprisal = self.create_sentence_window(curr_sent_ids, sentence, idx, right_idx, right_ctx)
                            wordpiece_surp['1SL/'+str(right_idx)+'R'].append((token, surprisal))

        return wordpiece_surp


def main(test=False):
    model = Surprisal(word_level=True, sentence_level=True)
    par = model.par_data
    sent = model.sent_data
    
    # For testing shorter text
    if test:
        par = {260: par.get(260)}
        sent = {260: sent.get(260)}
        #par = {260: ["God is an abstract concept indeed. Well I don't Know. What on earth are we doing here."]}
        #sent = {260: ["God is an abstract concept indeed.", "Well I don't Know.", "What on earth are we doing here."]}
    
    if model.word_level:
        surp_word = model.estimate_surprisal(par)
    if model.sentence_level:
        surp_sent = model.estimate_sentence_surprisal(sent)
    return surp_word, surp_sent


if __name__ == '__main__':  
    surp_word, surp_sent = main(test=True)

    data_word = []
    for key, surp in surp_word.items():
        df = pd.DataFrame(surp_word.get(key), columns=['WordPiece','surprisal'])
        df['window'] = key
        data_word.append(df)

    data_sent = []
    for key, surp in surp_sent.items():
        df = pd.DataFrame(surp_sent.get(key), columns=['WordPiece','surprisal'])
        df['window'] = key
        data_sent.append(df)

    surp_word_all = pd.concat(data_word, axis=0, ignore_index=True)
    surp_sent_all = pd.concat(data_sent, axis=0, ignore_index=True)

    #surp_word_all.to_csv('surprisal_paragraph_level.csv', sep='\t')
    #surp_sent_all.to_csv('surprisal_sentence_level.csv', sep='\t')
    
    sns.kdeplot(data=surp_word_all, x='surprisal', hue='window')
    plt.xlabel('Surprisal')
    plt.ylabel('Frequency')
    plt.title('Cross Surprisal Word Level - 1 Bible Paragraph')
    plt.savefig('cross_surp_word_new.png', dpi=300)
    plt.show()

    sns.kdeplot(data=surp_sent_all, x='surprisal', hue='window')
    plt.xlabel('Surprisal')
    plt.ylabel('Frequency')
    plt.title('Cross Surprisal Sentence Level - 1 Bible Paragraph')
    plt.savefig('cross_surp_sent_new.png', dpi=300)
    plt.show()