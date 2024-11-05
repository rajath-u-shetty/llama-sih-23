from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


class tok(object):
    def _init_(self):
        self.model = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
        self.tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate", src_lang="en_XX")
    

        

        

    def predict(self,article_en):
        model_inputs = self.tokenizer(article_en, return_tensors="pt")

        # translate from English to Hindi
        generated_tokens = self.model.generate(
            **model_inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id["hi_IN"]
        )
        self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

mod = tok()
print(mod.predict("hi"))