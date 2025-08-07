import torch.nn as nn

attribute_size={'100': 7, '117': 286, '111': 58735, '118': 746, '101': 45, '102': 11137, '119': 1746, '120': 1344, '114': 18, '112': 26, '121': 49080, '115': 188, '122': 11152, '116': 19}
attribute_embeddings = nn.ModuleDict({
    attr: nn.Embedding(size, 5, padding_idx=0)
    for attr, size in attribute_size.items()}
)  # TODO: 所有attr用一个hidden_size吗？

print(attribute_embeddings)
print("*"*10)
print(attribute_embeddings["100"])
