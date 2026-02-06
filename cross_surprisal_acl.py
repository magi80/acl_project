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



class Surprisal:
    def __init__(self, word_level=False, sentence_level=False):
        self.device =  torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = BertForMaskedLM.from_pretrained("bert-base-cased").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.ctx = 512
        self.word_level = word_level
        self.sentence_level = sentence_level
        print(f"--Using device: {self.device}") 


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
        att = torch.ones_like(new_input).unsqueeze(0).to(self.device)
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
        print(f"--Context Window: {left}L/{right}R | Masked Token: [{token}]")
        print(self.tokenizer.convert_ids_to_tokens(new_input[0].tolist()))
        return token, surprisal


    def create_sentence_window(self, prev_sent_ids, curr_sent_ids, right_ctx, s, r):
        """
        Create a context window at the sentence level with asymmetrical 
        right context: [CLS] full_sentence [MASK] 0_word / 1_word / 2/word / full_sentence [SEP]
        where the [MASK] token is always the 1st word of Ssentence Si+1 (Note that the 
        function's logic here need to be optimized). Returns a dictionary with the surprisal scores
        for each masked token.
        """
        window = [self.tokenizer.cls_token_id]
        window += prev_sent_ids
        window.append(self.tokenizer.mask_token_id)

        window = [self.tokenizer.cls_token_id]
        window += prev_sent_ids
        window.append(self.tokenizer.mask_token_id)
        window += right_ctx                              #curr_sent_ids[1:1+right]
        window.append(self.tokenizer.sep_token_id)

        new_input = torch.tensor(window).unsqueeze(0).to(self.device)
        att = torch.ones_like(new_input).unsqueeze(0).to(self.device)
        masked_input = {'input_ids': new_input, 'attention_mask': att}
        outputs = self.model(**masked_input)
        logits = outputs['logits']

        mask_pos = 1 + len(prev_sent_ids)
        probs = torch.softmax(logits[0, mask_pos], dim=-1)

        target = curr_sent_ids[0].item()
        print('--Current Mask:', self.tokenizer.convert_ids_to_tokens(target))
        token_prob = probs[target].item()
        surprisal = -math.log2(token_prob)

        token = self.tokenizer.convert_ids_to_tokens(target)
        # DEBUG
        print("-"*50)
        print(f"--Current Context Window for sentence {s}: SL/{r}R | Masked Token: [{token}]")
        print(self.tokenizer.convert_ids_to_tokens(new_input[0].tolist()))
        return token, surprisal


    def estimate_surprisal(self, input):
        """
        Estimates surpisal values at the word (subtoken) level
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

                    #tokenized = [sent_ids[1:-1] for sent_ids in inputs["input_ids"]]
                    # DEBUG
                    #print('--Input:', tokenized)

                for s in range(1, len(tokenized)):
                    prev_sent_ids = tokenized[s-1]
                    curr_sent_ids = tokenized[s]

                    print('--Previous sentence:')
                    print(self.tokenizer.convert_ids_to_tokens(prev_sent_ids))

                    for right in [0, 1, 2, "S"]:                        # 0 word, 1 word, 2 word from the next sentence (Si+1) plus full sentence (Si+1) 
                        if right == 'S':
                            right_ctx = curr_sent_ids[1:]               # Comprise the full next sentence (Si+1), only 1st word is masked
                        else:
                            if len(curr_sent_ids) <= right:
                                continue
                            right_ctx = curr_sent_ids[1:1+right]
                        
                        token, surprisal = self.create_sentence_window(prev_sent_ids, curr_sent_ids, right_ctx, s, right)
                        wordpiece_surp['SL/'+str(right)+'R'].append((token, surprisal))

        return wordpiece_surp


def main():
    features = ExtractFeatures()
    raw_bible_text = [bible.get('orth').get('txt') for bible in features.raw_orth]  # RAW bible text
    raw, raw_length = get_sentences_dct(raw_bible_text)                             # RAW TEXTS as dcts
    sent, sent_length = get_raw_sentences(raw_bible_text)                           # SENTENCES

    test_sentence = {'chap1': ["Paul a servant of God and God's death an apostle of Jesus Christ according to the faith of God's elect and the acknowledging of the truth which is after godliness in hope of eternal life which God that cannot lie promised before the world began.",
                               "God is here there and everywhere in the sky.",
                               "Behold you skyscraper of mustard."]}
    test_raw = {'chap:2': ["God is an abstract concept indeed. Well I don't Know. What on earth are we doing here."]}
    test = raw.get('chapter_259')
    test_paragraph = {'chapter_259': test}                                          # Use this to test one single paragraph

    test_s = sent.get('chapter_259')
    test_sent = {'chapter_259': test_s}

    print('--Total word for paragraph:', len(test.split()) )
    print('--Total sentences:', len(test_s))

    model = Surprisal(word_level=True, sentence_level=True)
    if model.word_level:
        surp_word = model.estimate_surprisal(test_paragraph)
    if model.sentence_level:
        surp_sent = model.estimate_sentence_surprisal(test_sent)
    return surp_word, surp_sent


def get_raw_sentences(sentences):
    """
    Converts a Bible paragraph into a dictionary of
    sentences for each .json file. Retruns a dictioanry
    where the keys are the file numbers while the values 
    are lists of strings, each string is a separate sentence.  
    """
    sent = {}
    sent_length = {}
    for i in range(len(sentences)):
        #print(sentences[i])
        #sen = ' '.join(sentences[i])
        sent[f'chapter_{i}'] = sentences[i] #ADDED sentence[i]
        sent_length[f'chapter_{i}'] = len(' '.join(sentences[i]).split())
    return sent, sent_length


if __name__ == '__main__':  
    surp_word, surp_sent = main()

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

    print(surp_word_all)
    
    sns.histplot(data=surp_word_all, x='surprisal', hue='window', bins=50, kde=True)
    plt.xlabel('Surprisal')
    plt.ylabel('Frequency')
    plt.title('Cross Surprisal Word Level - 1 Bible Paragraph')
    plt.savefig('cross_surp_word.png', dpi=300)
    plt.show()

    sns.histplot(data=surp_sent_all, x='surprisal', hue='window', bins=50, kde=True)
    plt.xlabel('Surprisal')
    plt.ylabel('Frequency')
    plt.title('Cross Surprisal Sentence Level - 1 Bible Paragraph')
    plt.savefig('cross_surp_sent.png', dpi=300)
    plt.show()