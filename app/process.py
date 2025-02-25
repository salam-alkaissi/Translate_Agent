from difflib import Differ
from sacrebleu import BLEU, CHRF, TER
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import docx
import gradio as gr
import pymupdf
from icecream import ic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from patch import (
    calculate_chunk_size,
    model_load,
    multichunk_improve_translation,
    multichunk_initial_translation,
    multichunk_reflect_on_translation,
    num_tokens_in_string,
    one_chunk_improve_translation,
    one_chunk_initial_translation,
    one_chunk_reflect_on_translation,
)
from simplemma import simple_tokenizer


progress = gr.Progress()


def extract_text(path):
    with open(path) as f:
        file_text = f.read()
    return file_text


def extract_pdf(path):
    doc = pymupdf.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_docx(path):
    doc = docx.Document(path)
    data = []
    for paragraph in doc.paragraphs:
        data.append(paragraph.text)
    content = "\n\n".join(data)
    return content


# Add to process.py
def create_error_plot(error_types: dict, target_lang: str):
    """Create a styled error analysis visualization"""
    fig = plt.figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    # Configure error types and colors
    # error_categories = {
    #     'additions': ('Additions', '#FF6B6B'),
    #     'deletions': ('Deletions', '#4ECDC4'),
    #     'changes': ('Changes', '#45B7D1')
    # }
    #  Prepare data
    labels = ['Additions', 'Deletions', 'Changes']
    values = [
        error_types['additions'],
        error_types['deletions'],
        error_types['changes']
    ]
    
    bars = ax.bar(labels, values, color=['#ff7f0e', '#1f77b4', '#2ca02c'])
    # Prepare data
    # labels = [error_categories[key][0] for key in error_types.keys()]
    # values = list(error_types.values())
    # colors = [error_categories[key][1] for key in error_types.keys()]
    
    # Create bars with value labels
    # bars = ax.bar(labels, values, color=colors, edgecolor='white')
    
    # Add value labels
    # for bar in bars:
    #     height = bar.get_height()
    #     ax.text(bar.get_x() + bar.get_width()/2., height,
    #             f'{height:.1f}',
    #             ha='center', va='bottom')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # Style the plot
    # ax.set_title(f'Error Analysis - {target_lang}', fontsize=14, pad=20)
    # ax.set_ylabel('Error Count', fontsize=12)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.grid(axis='y', alpha=0.3)
    
    # plt.tight_layout()
    
    ax.set_title(f'Error Analysis - {target_lang}')
    ax.set_ylabel('Error Count')
    plt.close(fig)
    # plt.tight_layout()
    
    return fig

def calculate_metrics(reference: str, candidate: str, source_lang: str, target_lang: str):
    """Calculate multiple translation quality metrics"""
    metrics = {}
    
    differ = Differ()
    diff = list(differ.compare(reference.split(), candidate.split()))
    
    # Tokenize for language-specific metrics
    ref_tokens = [reference.split()]
    cand_tokens = candidate.split()
    
    # BLEU Score (with effective_order for sentence-level)
    bleu = BLEU(effective_order=True)
    metrics['bleu'] = bleu.sentence_score(candidate, [reference]).score
    
    # # BLEU Score
    # bleu = BLEU()
    # metrics['bleu'] = bleu.sentence_score(candidate, [reference]).score
    
    # TER (Translation Edit Rate)
    ter = TER()
    metrics['ter'] = ter.sentence_score(candidate, [reference]).score
    
    # ChrF (Character F-score)
    chrf = CHRF()
    metrics['chrf'] = chrf.sentence_score(candidate, [reference]).score
    
    # METEOR
    metrics['meteor'] = meteor_score(ref_tokens, cand_tokens)
    
    # BERTScore
    _, _, f1 = bert_score([candidate], [reference], lang=target_lang)
    metrics['bert_score'] = f1.mean().item()
    
    # Add language-specific error analysis
    error_types = {
        'additions': sum(1 for d in diff if d.startswith('+ ')),
        'deletions': sum(1 for d in diff if d.startswith('- ')),
        'changes': sum(1 for d in diff if d.startswith('? '))
    }
    
    metrics['error_analysis'] = error_types
    metrics['error_plot'] = create_error_plot(error_types, target_lang)
    
    return metrics 

def tokenize(text):
    # Use nltk to tokenize the text
    words = simple_tokenizer(text)
    # Check if the text contains spaces
    if " " in text:
        # Create a list of words and spaces
        tokens = []
        for word in words:
            tokens.append(word)
            if not word.startswith("'") and not word.endswith(
                "'"
            ):  # Avoid adding space after punctuation
                tokens.append(" ")  # Add space after each word
        return tokens[:-1]  # Remove the last space
    else:
        return words


def diff_texts(text1, text2):
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    d = Differ()
    diff_result = list(d.compare(tokens1, tokens2))

    highlighted_text = []
    for token in diff_result:
        word = token[2:]
        category = None
        if token[0] == "+":
            category = "added"
        elif token[0] == "-":
            category = "removed"
        elif token[0] == "?":
            continue  # Ignore the hints line

        highlighted_text.append((word, category))

    return highlighted_text


# modified from src.translaation-agent.utils.tranlsate
def translator(
    source_lang: str,
    target_lang: str,
    source_text: str,
    country: str,
    max_tokens: int = 1000,
):
    """Translate the source_text from source_lang to target_lang."""
    num_tokens_in_text = num_tokens_in_string(source_text)

    ic(num_tokens_in_text)

    if num_tokens_in_text < max_tokens:
        ic("Translating text as single chunk")

        progress((1, 3), desc="First translation...")
        init_translation = one_chunk_initial_translation(
            source_lang, target_lang, source_text
        )

        progress((2, 3), desc="Reflection...")
        reflection = one_chunk_reflect_on_translation(
            source_lang, target_lang, source_text, init_translation, country
        )

        progress((3, 3), desc="Second translation...")
        final_translation = one_chunk_improve_translation(
            source_lang, target_lang, source_text, init_translation, reflection
        )

        return init_translation, reflection, final_translation

    else:
        ic("Translating text as multiple chunks")

        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        ic(token_size)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-3.5-turbo", #  "gpt-3.5-turbo" , "gpt-4"
            chunk_size=token_size,
            chunk_overlap=0,
        )

        source_text_chunks = text_splitter.split_text(source_text)

        progress((1, 3), desc="First translation...")
        translation_1_chunks = multichunk_initial_translation(
            source_lang, target_lang, source_text_chunks
        )

        init_translation = "".join(translation_1_chunks)

        progress((2, 3), desc="Reflection...")
        reflection_chunks = multichunk_reflect_on_translation(
            source_lang,
            target_lang,
            source_text_chunks,
            translation_1_chunks,
            country,
        )

        reflection = "".join(reflection_chunks)

        progress((3, 3), desc="Second translation...")
        translation_2_chunks = multichunk_improve_translation(
            source_lang,
            target_lang,
            source_text_chunks,
            translation_1_chunks,
            reflection_chunks,
        )

        final_translation = "".join(translation_2_chunks)

        return init_translation, reflection, final_translation


def translator_sec(
    endpoint2: str,
    base2: str,
    model2: str,
    api_key2: str,
    source_lang: str,
    target_lang: str,
    source_text: str,
    country: str,
    max_tokens: int = 1000,
):
    """Translate the source_text from source_lang to target_lang."""
    num_tokens_in_text = num_tokens_in_string(source_text)

    ic(num_tokens_in_text)

    if num_tokens_in_text < max_tokens:
        ic("Translating text as single chunk")

        progress((1, 3), desc="First translation...")
        init_translation = one_chunk_initial_translation(
            source_lang, target_lang, source_text
        )

        try:
            model_load(endpoint2, base2, model2, api_key2)
        except Exception as e:
            raise gr.Error(f"An unexpected error occurred: {e}") from e

        progress((2, 3), desc="Reflection...")
        reflection = one_chunk_reflect_on_translation(
            source_lang, target_lang, source_text, init_translation, country
        )

        progress((3, 3), desc="Second translation...")
        final_translation = one_chunk_improve_translation(
            source_lang, target_lang, source_text, init_translation, reflection
        )

        return init_translation, reflection, final_translation

    else:
        ic("Translating text as multiple chunks")

        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        ic(token_size)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-3.5-turbo", # "gpt-3.5-turbo", "gpt-4"
            chunk_size=token_size,
            chunk_overlap=0,
        )

        source_text_chunks = text_splitter.split_text(source_text)

        progress((1, 3), desc="First translation...")
        translation_1_chunks = multichunk_initial_translation(
            source_lang, target_lang, source_text_chunks
        )

        init_translation = "".join(translation_1_chunks)

        try:
            model_load(endpoint2, base2, model2, api_key2)
        except Exception as e:
            raise gr.Error(f"An unexpected error occurred: {e}") from e

        progress((2, 3), desc="Reflection...")
        reflection_chunks = multichunk_reflect_on_translation(
            source_lang,
            target_lang,
            source_text_chunks,
            translation_1_chunks,
            country,
        )

        reflection = "".join(reflection_chunks)

        progress((3, 3), desc="Second translation...")
        translation_2_chunks = multichunk_improve_translation(
            source_lang,
            target_lang,
            source_text_chunks,
            translation_1_chunks,
            reflection_chunks,
        )

        final_translation = "".join(translation_2_chunks)

        return init_translation, reflection, final_translation
    
