# import json
# import pickle
# import numpy as np
# import string
# from rank_bm25 import BM25Okapi
# from pyvi.ViTokenizer import tokenize
# from sentence_transformers import SentenceTransformer


# def split_text(text):
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     words = text.lower().split()
#     return [w for w in words if w.strip()]


# class Retriever:
#     def __init__(self, corpus, corpus_emb_path, model_name):
#         self.corpus = corpus
#         self.corpus_emb_path = corpus_emb_path
#         self.model_name = model_name

#         # Load model embedding
#         self.embedder = SentenceTransformer(model_name)

#         # Load embedding trực tiếp ở đây
#         with open(self.corpus_emb_path, "rb") as f:
#             self.embeddings = pickle.load(f)

#         # Normalize embeddings
#         self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
#         # BM25
#         # for doc in self.corpus:
#         #     if "passage" not in doc:
#         #         doc["passage"] = doc.get("context", "")
#         self.tokenized_corpus = [split_text(doc["passage"]) for doc in self.corpus]
#         self.bm25 = BM25Okapi(self.tokenized_corpus)

#     def retrieve(self, question: str, topk: int = 10):
#         segmented_question = tokenize(question)
#         query_emb = self.embedder.encode([segmented_question])[0]
#         query_emb = query_emb / np.linalg.norm(query_emb)

#         sim_scores = np.dot(self.embeddings, query_emb)

#         tokenized_query = split_text(question)
#         bm25_scores = self.bm25.get_scores(tokenized_query)
#         max_bm25, min_bm25 = max(bm25_scores), min(bm25_scores)
#         normalize = lambda x: (x - min_bm25 + 0.1) / (max_bm25 - min_bm25 + 0.1)

#         results = []
#         for i, doc in enumerate(self.corpus):
#             bm25_score = bm25_scores[i]
#             bm25_normed = normalize(bm25_score)
#             sem_score = sim_scores[i]
#             combined_score = bm25_normed * 0.4 + sem_score * 0.6
#             results.append({
#                 "id": doc["id"],
#                 "title": doc.get("title", ""),
#                 "context": doc.get("context", ""),
#                 "bm25_score": bm25_score,
#                 "semantic_score": sem_score,
#                 "combined_score": combined_score
#             })

#         return sorted(results, key=lambda x: x["combined_score"], reverse=True)[:topk]
import json
import pickle
import numpy as np
import string
from rank_bm25 import BM25Okapi
from pyvi.ViTokenizer import tokenize
from sentence_transformers import SentenceTransformer


def split_text(text):
    """
    Tách văn bản thành các từ và loại bỏ dấu câu, chuyển về chữ thường.
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.lower().split()
    return [w for w in words if w.strip()]


class Retriever:
    """
    Lớp Retriever để tìm kiếm thông tin bằng cách kết hợp semantic search (embeddings)
    và lexical search (BM25).
    """
    def __init__(self, corpus, corpus_emb_path, model_name):
        """
        Khởi tạo Retriever.

        Args:
            corpus (list): Danh sách các tài liệu (mỗi tài liệu là một dict).
            corpus_emb_path (str): Đường dẫn đến tệp pickle chứa embeddings của corpus.
            model_name (str): Tên hoặc đường dẫn đến mô hình SentenceTransformer.
        """
        self.corpus = corpus
        self.corpus_emb_path = corpus_emb_path
        self.model_name = model_name

        # Tải mô hình embedding từ SentenceTransformers
        try:
            self.embedder = SentenceTransformer(model_name)
            print(f"Đã tải mô hình SentenceTransformer: {model_name}")
        except Exception as e:
            print(f"LỖI: Không thể tải mô hình SentenceTransformer '{model_name}': {e}")
            raise # Re-raise the exception to stop execution


        # Tải embeddings của corpus từ tệp pickle
        self.embeddings = None # Khởi tạo an toàn
        try:
            print(f"Đang tải embeddings từ: {self.corpus_emb_path}")
            with open(self.corpus_emb_path, "rb") as f:
                loaded_embeddings_data = pickle.load(f)
            
            print(f"Dữ liệu embeddings tải về có kiểu: {type(loaded_embeddings_data)}")

            # Nếu dữ liệu tải về là list, kiểm tra xem các phần tử có phải là dict không
            # và trích xuất vector embedding thực sự từ key 'embedding'.
            if isinstance(loaded_embeddings_data, list):
                print("Dữ liệu là LIST. Đang kiểm tra cấu trúc bên trong và trích xuất embeddings.")
                extracted_embeddings = []
                for item in loaded_embeddings_data:
                    if isinstance(item, dict) and 'embedding' in item:
                        # Trích xuất numpy array từ key 'embedding'
                        extracted_embeddings.append(item['embedding'])
                    else:
                        # Nếu có phần tử không phải dict hoặc không có key 'embedding'
                        raise ValueError(f"Một phần tử trong list không phải dict hoặc không có key 'embedding': {item}")
                
                if extracted_embeddings:
                    self.embeddings = np.array(extracted_embeddings)
                    print(f"Đã chuyển đổi thành công list of dicts sang numpy array. Shape: {self.embeddings.shape}")
                else:
                    print("Cảnh báo: List embeddings tải về rỗng sau khi trích xuất hoặc không có embeddings hợp lệ.")
                    self.embeddings = np.array([]) # Tạo mảng numpy rỗng

            elif isinstance(loaded_embeddings_data, dict):
                # Nếu dữ liệu tải về là dict (trường hợp mà chúng ta đã từng nghĩ là lỗi trước đó)
                print("Dữ liệu là DICT. Đang cố gắng chuyển đổi values sang numpy array.")
                if loaded_embeddings_data:
                    first_value = next(iter(loaded_embeddings_data.values()))
                    if isinstance(first_value, (list, np.ndarray)):
                        self.embeddings = np.array(list(loaded_embeddings_data.values()))
                        print(f"Đã chuyển đổi thành công dict (values) sang numpy array. Shape: {self.embeddings.shape}")
                    else:
                        raise TypeError(f"Giá trị trong dict không phải list hoặc np.ndarray. Kiểu: {type(first_value)}")
                else:
                    print("Cảnh báo: Dict embeddings tải về rỗng.")
                    self.embeddings = np.array([]) # Tạo mảng numpy rỗng

            elif isinstance(loaded_embeddings_data, np.ndarray):
                print("Dữ liệu đã là NUMPY ARRAY. Sử dụng trực tiếp.")
                self.embeddings = loaded_embeddings_data
                print(f"Embeddings đã là numpy array. Shape: {self.embeddings.shape}")
            else:
                # Xử lý trường hợp kiểu dữ liệu không mong muốn
                raise TypeError(f"Dữ liệu embeddings tải từ '{self.corpus_emb_path}' có kiểu không hợp lệ: {type(loaded_embeddings_data)}. Expected dict, list, or np.ndarray.")

        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy tệp embeddings tại '{self.corpus_emb_path}'.")
            raise
        except Exception as e:
            print(f"LỖI: Khi tải hoặc xử lý tệp embeddings '{self.corpus_emb_path}': {e}")
            raise

        # --- Dòng code gây lỗi trước đó sẽ chạy SAU KHI self.embeddings được gán giá trị đúng ---
        print(f"\nKiểm tra self.embeddings NGAY TRƯỚC khi chuẩn hóa:")
        print(f"Kiểu của self.embeddings: {type(self.embeddings)}")
        if isinstance(self.embeddings, np.ndarray):
            print(f"Shape của self.embeddings: {self.embeddings.shape}")
            if self.embeddings.ndim == 1:
                print(f"Cảnh báo: Embeddings có dạng 1D. Điều này có thể không chính xác nếu embeddings có nhiều chiều (ví dụ 768).")
                print(f"Dữ liệu đầu tiên của self.embeddings: {self.embeddings[0]}")
            elif self.embeddings.ndim > 1 and self.embeddings.size > 0:
                print(f"Dữ liệu đầu tiên của self.embeddings: {self.embeddings[0][:5]} (chỉ 5 phần tử đầu của vector đầu tiên)")
            else:
                print("self.embeddings là một numpy array rỗng hoặc có shape không xác định.")
        elif self.embeddings is None:
            print("self.embeddings vẫn là None! Có lỗi trong quá trình tải trước đó.")
        else:
            print(f"self.embeddings vẫn không phải numpy array sau khi xử lý: {self.embeddings}")


        # Normalize embeddings
        if self.embeddings is not None and isinstance(self.embeddings, np.ndarray) and self.embeddings.size > 0:
            if self.embeddings.ndim < 2:
                print("LỖI CẤU TRÚC: Embeddings cần có ít nhất 2 chiều để chuẩn hóa với axis=1. Đang thử reshape.")
                try:
                    self.embeddings = self.embeddings.reshape(-1, 1)
                except ValueError as ve:
                    print(f"LỖI: Không thể reshape embeddings thành 2 chiều. Lỗi: {ve}")
                    raise
            
            if self.embeddings.shape[1] == 0:
                print("Cảnh báo: Kích thước chiều thứ hai của embeddings là 0. Không thể chuẩn hóa.")
                self.embeddings = np.array([]) 
            else:
                self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                print("Đã chuẩn hóa embeddings của corpus thành công.")
        else:
            print(f"Cảnh báo: Dữ liệu embeddings từ '{self.corpus_emb_path}' rỗng hoặc không đúng định dạng. Không thể chuẩn hóa.")
            self.embeddings = np.array([]) 


        # BM25
        self.tokenized_corpus = [split_text(doc.get("context", "")) for doc in self.corpus]
        
        if self.tokenized_corpus and any(len(x) > 0 for x in self.tokenized_corpus): 
             self.bm25 = BM25Okapi(self.tokenized_corpus)
             print("Đã khởi tạo BM25Okapi.")
        else:
            print("Cảnh báo: Corpus rỗng hoặc không có nội dung để tokenize cho BM25. BM25 sẽ không hoạt động.")
            self.bm25 = None 

    def retrieve(self, question: str, topk: int = 10):
        """
        Tìm kiếm các tài liệu liên quan dựa trên câu hỏi.
        """
        if not (self.embeddings is not None and self.embeddings.size > 0 and self.embeddings.ndim >= 2):
            print("Lỗi: Embeddings của corpus chưa được tải, rỗng hoặc không có cấu trúc 2D hợp lệ. Không thể thực hiện tìm kiếm semantic.")
            return [] 

        # Mã hóa câu hỏi thành embedding
        segmented_question = tokenize(question)
        query_emb = self.embedder.encode([segmented_question])[0] 

        # Chuẩn hóa embedding của câu hỏi
        norm_query_emb = np.linalg.norm(query_emb)
        if norm_query_emb > 1e-9: 
            query_emb = query_emb / norm_query_emb
        else:
            print("Cảnh báo: Embedding của câu hỏi là vector 0 (hoặc gần 0). Không thể chuẩn hóa, sử dụng vector gốc.")
            pass 

        # Tính điểm tương đồng cosine (semantic score)
        if query_emb.ndim == 1:
            sim_scores = np.dot(self.embeddings, query_emb)
        else:
            print("Cảnh báo: query_emb không phải 1D. Đang cố gắng tính np.dot trực tiếp. Có thể cần reshape query_emb.")
            sim_scores = np.dot(self.embeddings, query_emb.T if query_emb.ndim > 1 and query_emb.shape[0] != self.embeddings.shape[1] else query_emb) 

        # Tính điểm BM25
        tokenized_query = split_text(question)
        
        bm25_scores = []
        if self.bm25: 
            bm25_scores = self.bm25.get_scores(tokenized_query)
        else:
            bm25_scores = [0] * len(self.corpus)
            print("Cảnh báo: BM25 không hoạt động do lỗi khởi tạo trước đó. Sử dụng điểm BM25 = 0.")


        # Chuẩn hóa điểm BM25
        max_bm25, min_bm25 = 0, 0
        
        # --- Dòng này đã được sửa để xử lý numpy array hoặc list một cách an toàn hơn ---
        if isinstance(bm25_scores, np.ndarray) and bm25_scores.size > 0: 
            max_bm25, min_bm25 = np.max(bm25_scores), np.min(bm25_scores) # Dùng np.max/min cho numpy array
        elif isinstance(bm25_scores, list) and len(bm25_scores) > 0:
            max_bm25, min_bm25 = max(bm25_scores), min(bm25_scores)
        else:
            pass # bm25_scores rỗng, giữ nguyên max_bm25 và min_bm25 là 0
        
        if (max_bm25 - min_bm25 + 0.1) == 0:
            normalize_bm25 = lambda x: 0.5 
            print("Cảnh báo: Tất cả điểm BM25 đều như nhau. Điểm chuẩn hóa BM25 sẽ là 0.5.")
        else:
            normalize_bm25 = lambda x: (x - min_bm25 + 0.1) / (max_bm25 - min_bm25 + 0.1)


        results = []
        for i, doc in enumerate(self.corpus):
            if i >= len(bm25_scores) or i >= len(sim_scores):
                print(f"Cảnh báo: Index {i} vượt quá kích thước của bm25_scores hoặc sim_scores. Bỏ qua tài liệu này.")
                continue

            bm25_score = bm25_scores[i]
            bm25_normed = normalize_bm25(bm25_score) 
            sem_score = sim_scores[i]
            
            combined_score = bm25_normed * 0.4 + sem_score * 0.6
            
            results.append({
                "id": doc.get("id"), 
                "title": doc.get("title", ""),
                "context": doc.get("context", ""),
                "bm25_score": bm25_score,
                "semantic_score": sem_score,
                "combined_score": combined_score
            })

        return sorted(results, key=lambda x: x["combined_score"], reverse=True)[:topk]

