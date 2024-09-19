import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# model = ParlerTTSForConditionalGeneration.from_pretrained("/mydev/dataspeech/weights/parler-tts-mini-jenny-30H").to(device)
# tokenizer = AutoTokenizer.from_pretrained("/mydev/dataspeech/weights/parler-tts-mini-jenny-30H")
model = ParlerTTSForConditionalGeneration.from_pretrained("/mydev/parler-tts/output_ling").to(device)
tokenizer = AutoTokenizer.from_pretrained("/mydev/parler-tts/output_ling")

prompt = "我是中天新闻主播"
description = "Ling speaks at an average pace with an animated delivery in a very confined sounding environment with clear audio quality."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

set_seed(42)
# specify min length to avoid 0-length generations
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, min_length=10)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("chinese.wav", audio_arr, model.config.sampling_rate)

