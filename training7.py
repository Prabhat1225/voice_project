# file_1.py
import subprocess

print("Running training7.py")
# Call the next file
subprocess.run(["python", "evaluate8.py"])





# from huggingface_hub import notebook_login

# notebook_login()
model.config.use_cache = False
# from transformers import Seq2SeqTrainingArguments

# training_args = Seq2SeqTrainingArguments(
#     output_dir="/content/drive/MyDrive/speecht5/model",  # change to a repo name of your choice
#     per_device_train_batch_size=16,
#     gradient_accumulation_steps=2,
#     learning_rate=1e-5,
#     warmup_steps=500,
#     max_steps=10000,
#     gradient_checkpointing=True,
#     fp16=True,
#     evaluation_strategy="steps",
#     per_device_eval_batch_size=8,
#     save_steps=1000,
#     eval_steps=1000,
#     logging_steps=25,
#     report_to=["tensorboard"],
#     load_best_model_at_end=True,
#     greater_is_better=False,
#     label_names=["labels"],
#     push_to_hub=False,
# )

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="/content/drive/MyDrive/speecht5/model",  # change to a repo name of your choice
    per_device_train_batch_size=16,  # Can be adjusted based on your GPU capacity
    gradient_accumulation_steps=2,  # For handling larger batches virtually
    learning_rate=2e-5,  # Slightly higher learning rate for speech models
    warmup_steps=1000,  # Adjust based on dataset size, start with 1000
    max_steps=20000,  # Increase for larger datasets like Common Voice (20000-30000 steps)
    gradient_checkpointing=True,
    fp16=True,  # Mixed precision for faster training
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    save_steps=1000,  # Save model every 1000 steps
    eval_steps=1000,  # Evaluate the model every 1000 steps
    logging_steps=50,  # Log progress more frequently
    report_to=["tensorboard"],  # Can add WandB or other reporting options
    load_best_model_at_end=True,
    greater_is_better=False,  # Can switch based on your evaluation metric
    label_names=["labels"],  # The target transcription column
    push_to_hub=False,  # Optional if you're pushing to Hugging Face Hub
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
)

trainer.train()

