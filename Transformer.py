import tiktoken

gpt2 = tiktoken.get_encoding("gpt2")

encoder = tiktoken.Encoding(
    name = "encoder",
    pat_str = gpt2._pat_str,
    mergeable_ranks = gpt2._mergeable_ranks,
    special_tokens = {
        **gpt2._special_tokens,
        "<START>": len(gpt2._mergeable_ranks) +1,
        "<END>": len(gpt2._mergeable_ranks) +2,
        "<PAD>": len(gpt2._mergeable_ranks) +3
    }
)


print(encoder.decode(encoder.encode("Hola que tal estas")))