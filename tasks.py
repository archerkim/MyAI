import spacy
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.html import partition_html
from unstructured.documents import elements as el
import uuid
from PIL import Image
import io
from pathlib import Path
from langdetect import detect, LangDetectException
from ingestor import ingest_data_task
from celery_app import app


from config import SPACY_MODEL_EN, SPACY_MODEL_RU, \
                   IMAGE_STORE_DIR, \
                   ARCHIVE_SUCCESS_DIR, ARCHIVE_FAILED_DIR
NLP_MODELS = {}

def get_nlp_model(lang):
    if lang not in NLP_MODELS:
        print(f"[*] loading spaCy model for language: {lang}")
        model_name = ""
        if lang == 'en':
            model_name = SPACY_MODEL_EN
        elif lang == 'ru':
            model_name = SPACY_MODEL_RU
        # ... можно добавить другие языки ...
        else:
            raise ValueError(f"unsupported language: {lang}")

        NLP_MODELS[lang] = spacy.load(model_name)

    return NLP_MODELS[lang]

@app.task(name="tasks.parse_document")
def parse_document_task(file_path):
    print(f"[*] processing: {file_path}")
    success = False
    try:
        file_path_obj = Path(file_path)
        file_extension = file_path_obj.suffix.lower()

        # --- шаг 1: "умный dispatch" для парсинга ---
        if file_extension == ".pdf":
            elements = partition_pdf(filename=file_path, strategy="hi_res", extract_images_in_pdf=True, infer_table_structure=True)
        # elif file_extension == ".docx": ...
        elif file_extension == ".html":
            elements = partition_html(filename=file_path)
        else:
            elements = partition(filename=file_path)

        processed_data = []

        # --- шаг 2: обработка каждого элемента ---
        for element in elements:
            # --- обработка картинок ---
            if isinstance(element, el.Image):
            #     # важный момент: unstructured с extract_images_in_pdf может не вернуть Image, а просто текст-плейсхолдер
            #     # и сохранить картинку рядом. эта логика может потребовать доработки на основе реальных
            #     # результатов unstructured для твоих pdf. для начала оставим так, но держим в уме.
            #     # ... твой код для обработки Image ...
            #     image_bytes = getattr(element.metadata, 'image_data', None)
            #     if image_bytes is None:
            #         print(f"Элемент Image найден, но атрибут image_data пуст.")
            #
            #     unique_filename = f"{uuid.uuid4()}.png"
            #
            #     image = Image.open(io.BytesIO(image_bytes))
            #
            #     output_path = IMAGE_STORE_DIR / unique_filename
            #
            #     image.save(output_path, format='PNG')
            #
            #     print(f"[*] saved and converted image to: {output_path}")
            #
            #     processed_data.append({
            #        "type": "image",
            #        "path": str(output_path) # сохраняем путь как строку
            #     })
                 pass
            # --- обработка текста ---
            else:
                text = element.text
                if not text or not text.strip():
                    continue

                try:
                    # lang = detect(text[:500])
                    # nlp = get_nlp_model(lang)
                    nlp = spacy.load(SPACY_MODEL_EN)
                    doc = nlp(text)
                    for sentence in doc.sents:
                        sentence_data = {
                            "text": sentence.text,
                            "tokens": [
                                {"lemma": token.lemma_, "dep": token.dep_, "head_offset": token.head.i - token.i}
                                for token in sentence
                            ]
                        }
                        processed_data.append({"type": "text", "content": sentence_data})

                except (LangDetectException, ValueError) as e:
                    print(f"[-] skipping a text block due to language issue: {e}")
                    continue

        # --- шаг 3: отправка результата дальше ---
        if not processed_data:
            print("[!] document processed, but no data was extracted.")
        else:
            print(f"[*] extracted {len(processed_data)} elements.")
            ingest_data_task.delay(processed_data)

        success = True
    except Exception as e:
        print(f"[!] fatal error processing {file_path}: {e}")
        # здесь ошибка фатальная, так что success остается False
    finally:
        archive_file(file_path, success=success)
    return success

def archive_file(file_path, success):
    source_path = Path(file_path)
    if success:
        target_dir = ARCHIVE_SUCCESS_DIR
    else:
        target_dir = ARCHIVE_FAILED_DIR
    target_path = target_dir / source_path.name
    source_path.rename(target_path)