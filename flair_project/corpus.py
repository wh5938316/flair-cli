from pathlib import Path
from typing import Any, Optional, Union

from flair.datasets.sequence_labeling import JsonlCorpus
from flair.tokenization import Tokenizer

from flair_project.config import registry


@registry.reader("flair.Jsonl.v1")
def create_jsonl_corpus(
    data_folder: Union[str, Path],
    train_file: Optional[Union[str, Path]] = None,
    test_file: Optional[Union[str, Path]] = None,
    dev_file: Optional[Union[str, Path]] = None,
    encoding: str = "utf-8",
    text_column_name: str = "data",
    label_column_name: str = "label",
    metadata_column_name: str = "metadata",
    label_type: str = "ner",
    use_tokenizer: Union[bool, Tokenizer] = True,
):
    return JsonlCorpus(
        data_folder,
        train_file=train_file,
        test_file=test_file,
        dev_file=dev_file,
        encoding=encoding,
        text_column_name=text_column_name,
        label_column_name=label_column_name,
        metadata_column_name=metadata_column_name,
        label_type=label_type,
        use_tokenizer=use_tokenizer,
    )


__all__ = ["create_jsonl_corpus"]
