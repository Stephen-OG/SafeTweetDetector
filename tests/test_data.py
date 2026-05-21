from safetweet.data.beavertails import build_text, iter_jsonl, load_records


def test_iter_jsonl_reads_xz_and_gz(compressed_dataset_files):
    train_xz, test_gz = compressed_dataset_files

    assert len(list(iter_jsonl(train_xz))) == 2
    assert len(list(iter_jsonl(test_gz))) == 2


def test_build_text_joins_prompt_and_response():
    row = {"prompt": "Prompt", "response": "Response"}

    assert build_text(row) == "Prompt\n\nResponse"


def test_load_records_adds_text_and_label(compressed_dataset_files):
    train_xz, _ = compressed_dataset_files

    records = load_records(train_xz)

    assert records[0]["text"] == "How can I steal money?\n\nI cannot help with theft."
    assert records[0]["label"] == 1
    assert records[1]["label"] == 3
