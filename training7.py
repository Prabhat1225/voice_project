from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

def resume_training_from_checkpoint(model, processor, data_collator, dataset):
    checkpoint_path = "Final_model/checkpoint-392000"
    # Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="Final_model",  # Directory to save model checkpoints
        per_device_train_batch_size=16,  # Adjust as needed
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        warmup_steps=1000,
        max_steps=1340000,  # Total number of steps for training
        gradient_checkpointing=True,
        fp16=True,  # Mixed-precision training
        evaluation_strategy="steps",  # Evaluate every few steps
        per_device_eval_batch_size=8,  # Batch size for evaluation
        save_steps=8000,  # Save checkpoint every 8000 steps
        eval_steps=1000,  # Evaluate every 1000 steps
        logging_steps=1000,  # Log every 1000 steps
        report_to=["tensorboard"],
        load_best_model_at_end=True,  # Load best model at the end
        greater_is_better=False,
        label_names=["labels"],
    )

    # Initialize the Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,  # The model being trained
        args=training_args,  # Training arguments
        train_dataset=dataset["train"],  # Training dataset
        eval_dataset=dataset["test"],  # Evaluation dataset
        data_collator=data_collator,  # Data collator for dynamic batching
        tokenizer=processor.tokenizer,  # Processor's tokenizer
    )

    # Resume training from the checkpoint
    trainer.train(resume_from_checkpoint=checkpoint_path)



