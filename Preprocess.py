from loading_model1 import downloading_model
from clean_up2 import clean_up
# from speaker3 import speaker
from prepare_dataset5 import preapare
from collator6 import collator
from training7 import resume_training_from_checkpoint

if __name__ == "__main__":
    processor,model,dataset = downloading_model()

    dataset,tokenizer= clean_up(processor,dataset)

    # dataset,model=speaker(dataset,model)

    dataset,processor=preapare(processor,dataset,tokenizer)

    processor, data_collator, dataset = collator(dataset,processor,model)

    resume_training_from_checkpoint(model, processor,data_collator,dataset)




