import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)

from transformers.generation_utils import GenerationMixin
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer

NEUTRAL = 0
TOXIC = 1

class GediAdapter(GenerationMixin):
    def __init__(
        self,  
        paraphraser_name,
        gedi_name,
        clf_name,
        target=NEUTRAL, # todo: pass them in the generate method as well
        gedi_logit_coef=1, 
        max_id=None,
        debug=False,
        reg_alpha=0,
        ub=None,
        lb=None,
        untouchable_tokens=None,
        nearly_infinity = -1000,
        device="cpu"
    ):
        self.paraphraser = AutoModelForSeq2SeqLM.from_pretrained(paraphraser_name).to(device)
        self.paraphraser_tokenizer = AutoTokenizer.from_pretrained(paraphraser_name)
        self.gedi_model = AutoModelForCausalLM.from_pretrained(gedi_name).to(device)
        self.clf = RobertaForSequenceClassification.from_pretrained(clf_name).to(device) if clf_name else None
        self.clf_tokenizer = RobertaTokenizer.from_pretrained(clf_name) if clf_name else None
        self.target = target
        self.gedi_logit_coef = gedi_logit_coef
        self.max_id = max_id 
        self.debug = debug
        self.reg_alpha = reg_alpha
        self.ub = ub
        self.lb = lb
        self.logits = []
        self.untouchable_tokens = untouchable_tokens or []
        self.nearly_infinity = nearly_infinity
        self._device = device
        logger.info(f"Running GeDi Adapter on {device}")

    def _setup(self, gedi_name):
        # Hard code by authors
        import os
        self.paraphraser.resize_token_embeddings(len(self.paraphraser_tokenizer)) 
        # add gedi-specific parameters
        if os.path.exists(gedi_name):
            w = torch.load(gedi_name + '/pytorch_model.bin', map_location='cpu')
            self.gedi_model.bias = w['bias'].to(self._device)
            self.gedi_model.logit_scale = w['logit_scale'].to(self._device)
            del w
        else:
            self.gedi_model.bias = torch.tensor([[ 0.08441592, -0.08441573]]).to(self._device)
            self.gedi_model.logit_scale = torch.tensor([[1.2701858]]).to(self._device)

        logger.info(f"bias: {self.gedi_model.bias}, logit_scale: {self.gedi_model.logit_scale}")

        NEW_POS = self.paraphraser_tokenizer.encode('normal', add_special_tokens=False)[0]
        NEW_NEG = self.paraphraser_tokenizer.encode('toxic', add_special_tokens=False)[0]
        self.codes = {'gedi_pos': NEW_POS, 'gedi_neg': NEW_NEG}

    def show_correction(self, sm, logits, corrected, old_logits):
        if self.paraphraser_tokenizer:
            vals = sm.cpu().numpy()[0][0]
            lv = logits.cpu().numpy()[0]
            cv = corrected.cpu().numpy()[0]
            if self.max_id:
                vals = vals[:self.max_id]
                lv = lv[:self.max_id]
                cv = cv[:self.max_id]
            # the most upgraded and downgraded tokens
            #print('+', self.paraphraser_tokenizer.convert_ids_to_tokens(np.argsort(-vals)[:5]), -np.sort(-vals)[:3])
            #print('-', self.paraphraser_tokenizer.convert_ids_to_tokens(np.argsort(vals)[:5]), np.sort(vals)[:3])
            #print(torch.exp(logits).sum())
            # how the top logits change
            old_top_id = np.argsort(-lv)[:5]
            new_top_id = np.argsort(-cv)[:5]
            toks = list(new_top_id)
            for t in old_top_id:
                if t not in toks:
                    toks.append(t)
            
            pos_logits = old_logits[0][0].cpu().numpy()
            neg_logits = old_logits[1][0].cpu().numpy()
        
            texts = self.paraphraser_tokenizer.convert_ids_to_tokens(toks)
            print('changes in the top:')
            for text, idx in zip(texts, toks):
                print('{:6d}: {:+2.2f} > {:+2.2f} {:20s}     [{:+2.2f} | {:+2.2f}]'.format(idx, lv[idx], cv[idx], text, pos_logits[idx], neg_logits[idx]))
            #print()
            #print(self.paraphraser_tokenizer.convert_ids_to_tokens(old_top_id), self.paraphraser_tokenizer.convert_ids_to_tokens(new_top_id))
        
    def __call__(self, return_dict=True, **kwargs):
        new_args = kwargs.get('main', {})
        with torch.no_grad():
            #print('main inputs:', {k: v.shape for k, v in new_args.items() if isinstance(v, torch.Tensor)})
            outputs = self.paraphraser(return_dict=return_dict, **new_args)
            #print('main outputs:', {k: v.shape for k, v in outputs.items() if isinstance(v, torch.Tensor)})
        outputs['main'] = outputs
        gedi_logits = {}
        for gedi_key in ['gedi_pos', 'gedi_neg']:
            gedi_args = kwargs.get(gedi_key, {})
            with torch.no_grad():
                gedi_out = self.gedi_model(**gedi_args, return_dict=True)
            outputs[gedi_key] = gedi_out  # logits are [batch, seq, voc]
            gedi_logits[gedi_key] = gedi_out['logits'][:,-1] # [batch, voc]
        # todo: add cumulative log loss
        stacked = torch.stack([gedi_logits['gedi_pos'], gedi_logits['gedi_neg']])  # [2, batch, voc]
        # exclude untouchable tokens from the distribution on which the penalty is calculated
        for token_id in self.untouchable_tokens:
            stacked[:, :, token_id] = self.nearly_infinity
        if self.reg_alpha:
            # increase each p(token|class) by the same amount, to shift odds ratio to 1.
            old_logits = torch.log(torch.softmax(stacked, -1) + self.reg_alpha)
        else:
            old_logits = torch.log_softmax(stacked, -1)
        
        if hasattr(self.gedi_model, 'logit_scale'):
            old_logits += self.gedi_model.logit_scale
        if hasattr(self.gedi_model, 'bias'):
            old_logits += self.gedi_model.bias.reshape(2, 1, 1).repeat(1,1,old_logits.shape[-1])

        sm = torch.log_softmax(old_logits, 0)
        logits = outputs['logits'][:,-1]
        
        # bad patch for eos
        #sm[:, :,1] = 0
        
        shift = sm[self.target]
        # shift everything by a constant to make logits before and after change more comparable 
        shift -= shift.mean() 
        # limit the positive or negative impact of gedi correction
        if self.lb is not None or self.ub is not None:
            shift = torch.clamp(shift, self.lb, self.ub)
        for token_id in self.untouchable_tokens:
            shift[:, token_id] = 0
        
        corrected = logits + shift * self.gedi_logit_coef
        if self.debug:
            self.show_correction(sm, logits, corrected, torch.log_softmax(stacked, -1))
        if self.max_id is not None:
            corrected[self.max_id:] = -np.infty
        outputs['logits'] = corrected.unsqueeze(1)  # add back sequence length
        # todo: calculate the loss with respect to the new logits
        
        #self.logits.append(corrected)
        return outputs

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # todo: prepare inputs separately for main and gedi models
        result = {}
        # unpack past after beam search application
        past = kwargs.get('past')
        #print('past:', past is not None)
        if past and not isinstance(past, tuple):
            for k, v in past.items():
                kwargs[k]['past'] = v
        
        main_input_ids = input_ids
        main_kwargs = kwargs.get('main', kwargs)
        if kwargs.get('main_prefix') is not None and main_kwargs.get('past') is None:
            prefix = kwargs['main_prefix'].unsqueeze(0).repeat(main_input_ids.shape[0], 1)
            main_input_ids = torch.cat([prefix, main_input_ids], dim=1)
            #print('main reshape text:', input_ids.shape, main_input_ids.shape)
            if main_kwargs.get('attention_mask') is not None:
                old_mask = main_kwargs['attention_mask']
                mask_prefix = prefix * 0 + 1
                main_kwargs['attention_mask'] = torch.cat([mask_prefix, old_mask], dim=1)
                #print('main reshape mask:', old_mask.shape, main_kwargs['attention_mask'].shape)
        #print('main    ', self.paraphraser_tokenizer.decode(main_input_ids[0]))
        result['main'] = self.paraphraser.prepare_inputs_for_generation(main_input_ids, **main_kwargs)
        
        for k in ['gedi_pos', 'gedi_neg']:
            gedi_args = kwargs.get(k, {})
            # todo: allow concatenation of codes to the input instead of replacing the first token
            if kwargs.get('gedi_prepend'):
                # prepend the code to the input
                prefix = torch.ones([input_ids.shape[0], 1], dtype=input_ids.dtype).to(input_ids.device) * self.codes[k]
                new_input_ids = torch.cat([prefix, input_ids], dim=1)
                # todo: enlarge the attention mask
            else:
                # instert the code instead of the first token of the input
                new_input_ids = input_ids.clone()  # batch size x seq len
                new_input_ids[:, 0] = self.codes[k]
            #print(k, self.paraphraser_tokenizer.decode(new_input_ids[0]))
            gedi_inputs = self.gedi_model.prepare_inputs_for_generation(new_input_ids, **gedi_args)
            result[k] = gedi_inputs
        return result

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False):
        # todo: use the new outputs as args
        result = {k: v for k, v in model_kwargs.items()}
        #print('main before upd:', {k: v.shape for k, v in model_kwargs.get('main', model_kwargs).items() if isinstance(v, torch.Tensor)}, 'main' in model_kwargs)
        result['main'] = self.paraphraser._update_model_kwargs_for_generation(
            outputs=outputs['main'], 
            model_kwargs=model_kwargs.get('main', model_kwargs), 
            is_encoder_decoder=self.paraphraser.config.is_encoder_decoder,
        )
        #print('main after upd:', {k: v.shape for k, v in result['main'].items() if isinstance(v, torch.Tensor)})
        for k in ['gedi_pos', 'gedi_neg']:
            result[k] = self.gedi_model._update_model_kwargs_for_generation(
                outputs=outputs[k], 
                model_kwargs=model_kwargs.get(k, {}), 
                is_encoder_decoder=self.gedi_model.config.is_encoder_decoder,
            )
           
        # a fix for beam search
        result['past'] = {
            k: result[k]['past'] 
            for k in ['main', 'gedi_pos', 'gedi_neg'] 
            if 'past' in result[k] and result[k]['past'] is not None and result[k]['past'][0] is not None
        }
        return result
    
    def _cleanup_gpu_ram(self):
        if torch.cuda.is_available() and self._device != 'cpu':
            import gc
            from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
            def get_gpu_utilization():
                nvmlInit()
                handle = nvmlDeviceGetHandleByIndex(0)
                info = nvmlDeviceGetMemoryInfo(handle)
                return info.used//1024**2

            logger.info(f"GPU memory occupied BEFORE cleanup: {get_gpu_utilization()} MB.")
            gc.collect()
            with torch.cuda.device(self._device):
                torch.cuda.empty_cache()
            logger.info(f"GPU memory occupied AFTER cleanup: {get_gpu_utilization()} MB.")

    def paraphrase(self, texts, max_length=128, beams=10):
        from text_processing import text_preprocess, text_postprocess
        texts = [text_preprocess(t) for t in texts]
        inputs = self.paraphraser_tokenizer(texts, return_tensors='pt', padding=True)['input_ids'].to(self._device)
        if max_length == 'auto':
            max_length = min(int(inputs.shape[1] * 1.1) + 4, 64)
        out = self.generate(
            inputs, 
            num_beams=beams,
            num_return_sequences=beams, 
            do_sample=False, 
            temperature=1.0, 
            repetition_penalty=3.0, 
            max_length=max_length,
            bad_words_ids=[[2]],  # unk
            output_scores=True, 
            return_dict_in_generate=True,
        )
        gen_texts = [self.paraphraser_tokenizer.decode(r, skip_special_tokens=True) for r in out.sequences]
        # self._cleanup_gpu_ram()
        return [text_postprocess(t) for t in gen_texts]
    
    def _predict_toxicity(self, texts):
        with torch.inference_mode():
            inputs = self.clf_tokenizer(texts, return_tensors='pt', padding=True)
            out = torch.softmax(self.clf(**inputs).logits, -1)[:, 1].cpu().numpy()
        return out

    def paraphrase_and_rerank(
        self,
        texts, 
        max_length='auto', 
        beams=10, 
        rerank=False, 
    ):
        assert isinstance(texts, list)
        gen_texts = self.paraphrase(texts=texts, max_length=max_length, beams=beams)
        if rerank:
            toxic_scores = self._predict_toxicity(gen_texts)
        
        out_texts = []
        for i in range(len(texts)):
            if rerank:
                idx = toxic_scores[(i*beams):((i+1)*beams)].argmin()
            else:
                idx = 0 # Results ordered from most likely to least
            out_texts.append(gen_texts[i*beams+idx])
        return out_texts
    
    def _reorder_cache(self, past, beam_idx):
        # for each model, cache should be reordered separately
        result = {}
        for key, subpast in past.items():
            model = self.paraphraser if key == 'main' else self.gedi_model
            result[key] = model._reorder_cache(subpast, beam_idx)
        return result
    
    # the data below is used only for compatibility
    # todo: clarify it
    def get_output_embeddings(self):
        return True

    @property
    def config(self):
        return self.paraphraser.config

    def get_encoder(self):
        return self.paraphraser.get_encoder()

    def parameters(self):
        return self.paraphraser.parameters()

    @property
    def device(self):
        return self.paraphraser.device

    @property
    def main_input_name(self):
        return self.paraphraser.main_input_name

    def forward(self, attention_mask=None, **kwargs):
        pass
