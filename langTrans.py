from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS


from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# article_en = ""
model = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate", src_lang="en_XX")

# model_inputs = tokenizer(article_en, return_tensors="pt")

# # translate from English to Hindi
# generated_tokens = model.generate(
#     **model_inputs,
#     forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
# )
# tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

# print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))


pipe = pipeline("translation", model="t5-large")


app = Flask(__name__)
CORS(app)  

@app.route("/")
def hello():
  return {"Server": "Active"}


@app.route('/translate', methods=['POST'])
def post_example():
    # global article_en
    data = request.get_json()
    print(data)
    text_input = data['text']

    model_inputs = tokenizer(text_input, return_tensors="pt")

    # translate from English to Hindi
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[data['target_language']]
    )
    tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

    output = pipe(text_input) #change parameters
    return {"translated_text": tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)}
    

    
if __name__ == '__main__':
    app.run()


# print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
# article_en = str(input())
