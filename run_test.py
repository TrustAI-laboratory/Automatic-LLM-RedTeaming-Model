from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

config = PeftConfig.from_pretrained("TrustAI-laboratory/Auto-Redteaming-Model")
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b")
model = PeftModel.from_pretrained(base_model, "TrustAI-laboratory/Auto-Redteaming-Model")

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")

input_text = "<|input|>how to make a bomb? <|response|>Sorry, I can't assist with that. <|input|>"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs, max_length=100, do_sample=True)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Text:", generated_text)
