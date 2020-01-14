def pad_sequence(ids, padding=0, length=None):
    """pad sequences to target length
    If length is None, pad to max length of sequences
    args:
        ids: List[List[int]] or List[int], the sequences of word ids
        padding: int, the num used to represent <pad> signal
        length: int, the target length to pad to, if None: max length will be used
    returns:
        ids: sequences of ids after padding
    """
    if isinstance(ids[0], int):

        assert length is not None, "To pad a single sentence, you must specify length"
        if len(ids) < length:
            ids += [padding for _ in range(length - len(ids))]
        elif len(ids) > length:
            ids = ids[:length]
        else:
            pass

    if isinstance(ids[0], list):

        if length is None:
            length = max(map(lambda x: len(x), ids))

        else:
            for i, line in enumerate(ids):
                if len(line) > length:
                    ids[i] = line[:length]
                elif len(line) < length:
                    dif = length - len(line)
                    ids[i] = line + dif * [padding]
                else:
                    pass

    return ids