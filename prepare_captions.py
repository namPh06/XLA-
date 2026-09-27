"""Prepare reproducible caption JSONL, with POS spans masked for SMP."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from transformers import CLIPTokenizer

SOURCES = ['dangne/gcc_caption_only', 'FredZhang7/stable-diffusion-prompts-2.47M']


def mask_caption(doc):
    spans, start, end = [], None, None
    for token in doc:
        if token.pos_ in {'NOUN', 'PROPN', 'ADJ', 'DET'}:
            start = token.idx if start is None else start
            end = token.idx + len(token.text)
        elif start is not None:
            spans.append((start, end))
            start = None
    if start is not None:
        spans.append((start, end))
    if not spans:
        words = [token for token in doc if not token.is_space and not token.is_punct]
        if not words:
            return None
        token = words[0]
        spans = [(token.idx, token.idx + len(token.text))]
    text = doc.text
    for start, end in reversed(spans):
        text = text[:start] + '$' + text[end:]
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/captions.jsonl')
    parser.add_argument('--per-source', type=int, default=50000)
    parser.add_argument('--input', help='Optional local UTF-8 file, one caption per line.')
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()
    if args.per_source < 1:
        parser.error('--per-source must be positive')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'{output} already exists; use a different output to preserve provenance.')
    import spacy
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'lemmatizer'])
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    sources = [args.input] if args.input else SOURCES
    seen, examples, counts = set(), [], {}
    for source in sources:
        if args.input:
            stream = ({'text': line} for line in Path(source).read_text(encoding='utf-8').splitlines())
        else:
            from datasets import load_dataset
            stream = load_dataset(source, split='train', streaming=True).shuffle(seed=args.seed, buffer_size=10000)
        captions = []
        for item in stream:
            raw = item['text']
            if not isinstance(raw, str):
                continue
            raw = raw.replace('<PERSON>', 'person').replace('$', '').lower().strip()
            token_ids = tokenizer.encode(raw, add_special_tokens=False, truncation=True, max_length=75)
            text = tokenizer.decode(token_ids).strip()
            if text and text not in seen:
                seen.add(text)
                captions.append(text)
            if len(captions) >= args.per_source:
                break
        count = 0
        for doc in nlp.pipe(captions, batch_size=128):
            masked = mask_caption(doc)
            if masked is not None:
                examples.append({'caption': doc.text, 'masked': masked, 'source': source})
                count += 1
        counts[source] = count
        print(f'{source}: {count} usable captions', flush=True)
    if not examples:
        raise ValueError('No usable captions; check the input.')
    random.Random(args.seed).shuffle(examples)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        for example in examples:
            stream.write(json.dumps(example, ensure_ascii=False) + '\n')
    metadata = {**vars(args), 'counts': counts, 'total': len(examples),
                'sha256': hashlib.file_digest(output.open('rb'), 'sha256').hexdigest(),
                'spacy': spacy.__version__, 'pos_model': nlp.meta['version'],
                'note': 'Pilot subset, deduplicated; not full paper training data.'}
    output.with_suffix('.meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
