import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
TRAIN_FILE = "../input/imdb.csv"
BERT_PATH = "../input/bert-model-uncased/"
EPOCHS = 10
ACCUMULATION = 2 #What is it?
MODEL_PATH = "model.bin"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)