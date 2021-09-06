---
language: th
datasets:
- common_voice
tags:
- audio
- automatic-speech-recognition
- speech
- xlsr-fine-tuning
license: cc-by-sa 4.0
---

# `wav2vec2-large-xlsr-53-th`
Finetuning `wav2vec2-large-xlsr-53` on Thai [Common Voice 7.0](https://commonvoice.mozilla.org/en/datasets)

[Read more on our blog](https://medium.com/airesearch-in-th/airesearch-in-th-3c1019a99cd)

We finetune [wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) based on [Fine-tuning Wav2Vec2 for English ASR](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_tuning_Wav2Vec2_for_English_ASR.ipynb) using Thai examples of [Common Voice Corpus 7.0](https://commonvoice.mozilla.org/en/datasets). The notebooks and scripts can be found in [vistec-ai/wav2vec2-large-xlsr-53-th](https://github.com/vistec-ai/wav2vec2-large-xlsr-53-th). The pretrained model and processor can be found at [airesearch/wav2vec2-large-xlsr-53-th](https://huggingface.co/airesearch/wav2vec2-large-xlsr-53-th).

## Usage

```
#load pretrained processor and model
processor = Wav2Vec2Processor.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")
model = Wav2Vec2ForCTC.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")

#function to resample to 16_000
def speech_file_to_array_fn(batch, 
                            text_col="sentence", 
                            fname_col="path",
                            resampling_to=16000):
    speech_array, sampling_rate = torchaudio.load(batch[fname_col])
    resampler=torchaudio.transforms.Resample(sampling_rate, resampling_to)
    batch["speech"] = resampler(speech_array)[0].numpy()
    batch["sampling_rate"] = resampling_to
    batch["target_text"] = batch[text_col]
    return batch

#get 2 examples as sample input
test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset["speech"][:2], sampling_rate=16_000, return_tensors="pt", padding=True)

#infer
with torch.no_grad():
    logits = model(inputs.input_values,).logits

predicted_ids = torch.argmax(logits, dim=-1)

print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", test_dataset["sentence"][:2])

>> Prediction: ['และ เขา ก็ สัมผัส ดีบุก', 'คุณ สามารถ รับทราบ เมื่อ ข้อความ นี้ ถูก อ่าน แล้ว']
>> Reference: ['และเขาก็สัมผัสดีบุก', 'คุณสามารถรับทราบเมื่อข้อความนี้ถูกอ่านแล้ว']
```

## Datasets

Common Voice Corpus 7.0](https://commonvoice.mozilla.org/en/datasets) contains 133 validated hours of Thai (255 total hours) at 5GB. We pre-tokenize with `pythainlp.tokenize.word_tokenize`. We preprocess the dataset using cleaning rules described in `notebooks/cv-preprocess.ipynb` by [@tann9949](https://github.com/tann9949). We then deduplicate and split as described in [ekapolc/Thai_commonvoice_split](https://github.com/ekapolc/Thai_commonvoice_split) in order to 1) avoid data leakage due to random splits after cleaning in [Common Voice Corpus 7.0](https://commonvoice.mozilla.org/en/datasets) and 2) preserve the majority of the data for the training set. The dataset loading script is `scripts/th_common_voice_70.py`. You can use this scripts together with `train_cleand.tsv`, `validation_cleaned.tsv` and `test_cleaned.tsv` to have the same splits as we do. The resulting dataset is as follows:

```
DatasetDict({
    train: Dataset({
        features: ['path', 'sentence'],
        num_rows: 86586
    })
    test: Dataset({
        features: ['path', 'sentence'],
        num_rows: 2502
    })
    validation: Dataset({
        features: ['path', 'sentence'],
        num_rows: 3027
    })
})
```

## Training

We fintuned using the following configuration on a single V100 GPU and chose the checkpoint with the lowest validation loss. The finetuning script is `scripts/wav2vec2_finetune.py`

```
# create model
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)
model.freeze_feature_extractor()
training_args = TrainingArguments(
    output_dir="../data/wav2vec2-large-xlsr-53-thai",
    group_by_length=True,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=16,
    metric_for_best_model='wer',
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_strategy="steps",
    logging_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    num_train_epochs=100,
    fp16=True,
    learning_rate=1e-4,
    warmup_steps=1000,
    save_total_limit=3,
    report_to="tensorboard"
)
```

## Evaluation

We benchmark on the test set using WER with words tokenized by [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp) 2.3.1 and [deepcut](https://github.com/rkcosmos/deepcut), and CER. We also measure performance when spell correction using [TNC](http://www.arts.chula.ac.th/ling/tnc/) ngrams is applied. Evaluation codes can be found in `notebooks/wav2vec2_finetuning_tutorial.ipynb`. Benchmark is performed on `test-unique` split.

|                                | WER PyThaiNLP 2.3.1 | WER deepcut    | CER            |
|--------------------------------|---------------------|----------------|----------------|
| [Kaldi from scratch](https://github.com/vistec-AI/commonvoice-th)         | 23.04              |                | 7.57         |
| Ours without spell correction  | 13.634024          | **8.152052** | **2.813019** |
| Ours with spell correction     | 17.996397          | 14.167975     | 5.225761     |
| Google Web Speech API※         | 13.711234          | 10.860058     | 7.357340     |
| Microsoft Bing Speech API※     | **12.578819**      | 9.620991     | 5.016620     |
| Amazon Transcribe※             | 21.86334           | 14.487553     | 7.077562     |
| NECTEC AI for Thai Partii API※ | 20.105887          | 15.515631     | 9.551027     |

※ APIs are not finetuned with Common Voice 7.0 data

## License

Models are under CC-BY-SA 4.0. All codes are under Apache 2.0.

## Ackowledgements
* model training and validation notebooks/scripts [@cstorm125](https://github.com/cstorm125/)
* dataset cleaning scripts [@tann9949](https://github.com/tann9949)
* dataset splits [@ekapolc](https://github.com/ekapolc/) and [@14mss](https://github.com/14mss)
* running the training [@mrpeerat](https://github.com/mrpeerat)
* spell correction [@wannaphong](https://github.com/wannaphong)

