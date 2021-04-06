# from tokenizers import BertWordPieceTokenizer

# # Initialize an empty BERT tokenizer
# tokenizer = BertWordPieceTokenizer(
#   clean_text=False,
#   handle_chinese_chars=False,
#   strip_accents=False,
#   lowercase=True,
# )

# prepare text files to train vocab on them
files = ['/home/go_team/parler_sampled.txt']

# # train BERT tokenizer
# tokenizer.train(
#   files,
#   vocab_size=70000,
#   min_frequency=1,
#   show_progress=True,
#   special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
#   limit_alphabet=1000,
#   wordpieces_prefix="##"
# )

# # save the vocab
# tokenizer.save('4chan_vocab.txt')

from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.models import BPE
import json

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train(files, trainer)
tokenizer.save("/home/go_team/parler_tokenizer.json")


path = "/home/go_team/parler_tokenizer.json"
master_string = ""
with open(path) as fp:
  for line in fp:
    master_string += line
conf = json.loads(master_string)
vocab = list(conf['model']['vocab'].keys())
print(type(vocab))
print(len(vocab))
with open("parler_vocab.txt", "w") as f:
  for item in vocab:
    f.write("%s\n" % item)