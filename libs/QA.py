from haystack.document_stores import ElasticsearchDocumentStore
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import FARMReader, EmbeddingRetriever
try:
        from haystack.nodes import BM25Retriever
except Exception:  # noqa: BLE001
        BM25Retriever = None

import os
import time
from pathlib import Path
from haystack.schema import Document
try:
        import torch
except Exception:  # noqa: BLE001
        torch = None

# Don't force a specific GPU; allow caller to control this.
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "")

host = os.getenv("ELASTICSEARCH_HOST", "127.0.0.1")
port = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
index = os.getenv("ELASTICSEARCH_INDEX", "once")
embedding_model = os.getenv("QA_EMBEDDING_MODEL", "timpal0l/mdeberta-v3-base-squad2")

_default_reader = os.getenv("QA_READER_MODEL", "models/deberta_iapp_lst20_CMSK_model")  # p'knot
model_name = _default_reader
if os.path.exists(_default_reader) is False:
        # Fallback so QA can run without downloading the custom model folder.
        model_name = os.getenv("QA_READER_MODEL_FALLBACK", "timpal0l/mdeberta-v3-base-squad2")
try:
        threshold = float(os.getenv("QA_THRESHOLD", "0.10"))
except ValueError:
        threshold = 0.10

_use_gpu_env = os.getenv("QA_USE_GPU", "").strip().lower() in {"1", "true", "yes"}
use_gpu = bool(_use_gpu_env and torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available())

_BASE_DIR = Path(__file__).resolve().parents[1]
data_dir = _BASE_DIR / "context"  # folder ที่เก็บ context

print("preparing qa model")
got_docs = []
for txt_path in sorted(Path(data_dir).glob("*.txt")):
        try:
                content = txt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
                content = txt_path.read_text(encoding="utf-8-sig")
        got_docs.append(Document(content=content, meta={"name": txt_path.name}))

document_store = ElasticsearchDocumentStore(
        host=host,
        port=port,
        index=index,
        analyzer="thai",
        similarity="dot_product",
        embedding_dim=768,
        timeout=30,
)  # embedding_dim ตามโมเดลที่เราใช้
# document_store = ElasticsearchDocumentStore(host="190.92.214.247",index="text_dataset",analyzer="thai",similarity="dot_product",embedding_dim=768)

# Avoid duplicate docs if the server is restarted.
try:
        document_store.delete_documents(index=index)
except Exception:
        pass

document_store.write_documents(got_docs)

retriever_kind = os.getenv("QA_RETRIEVER", "bm25").strip().lower()
if retriever_kind == "bm25" and BM25Retriever is not None:
        retriever = BM25Retriever(document_store=document_store)
else:
        retriever = EmbeddingRetriever(
                document_store=document_store,
                max_seq_len=512,
                progress_bar=False,
                embedding_model=embedding_model,
        )
        # sentence-transformers/clip-ViT-B-32-multilingual-v1 512
        # sentence-transformers/distiluse-base-multilingual-cased-v2  512
        document_store.update_embeddings(retriever, index, batch_size=8)

reader=FARMReader(model_name_or_path=model_name,use_gpu=use_gpu, max_seq_len=512, doc_stride=100, batch_size=8,progress_bar=False)
pipe = ExtractiveQAPipeline(reader, retriever)

print("preparing qa model ok")

def predict(quest, ret_tk=3, red_tk=1):
        start = time.perf_counter()
        prediction = pipe.run(
                        query=quest, params={"Retriever": {"top_k": ret_tk}, "Reader": {"top_k": red_tk}}
        )
        elapsed_s = time.perf_counter() - start
        print(f"QA.predict took {elapsed_s:.2f}s (ret_tk={ret_tk}, red_tk={red_tk})")

        answers = prediction.get('answers') or []
        if not answers:
                docs = prediction.get("documents") or []
                if docs:
                        preview = (docs[0].content or "").replace("\n", " ")[:160]
                        print(f"QA: no answers. Top doc preview: {preview!r}")
                return None

        best = answers[0]
        try:
                ans_preview = (best.answer or "").replace("\n", " ")[:160]
                print(f"QA: best score={best.score:.4f} answer={ans_preview!r}")
        except Exception:
                pass
        return best.answer if best.score >= threshold else None