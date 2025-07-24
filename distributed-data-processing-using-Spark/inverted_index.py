from pyspark.sql import SparkSession
import time
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
import os
import re

# Khởi tạo SparkSession
spark = SparkSession.builder \
    .appName("InvertedIndex") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Get default parallelism (usually based on total cores)
default_parallelism = spark.sparkContext.defaultParallelism
print(f"Default parallelism: {default_parallelism}")

# Get executor information
executor_count = len(spark.sparkContext._jsc.sc().statusTracker().getExecutorInfos()) - 1  # -1 for driver
print(f"Number of executors: {executor_count}")

# Get configuration info
executor_cores = spark.sparkContext.getConf().get("spark.executor.cores", "unknown")
print(f"Cores per executor: {executor_cores}")

# Bắt đầu đo thời gian tổng thể
total_start_time = time.perf_counter()

# Load NLTK resources on driver node only
nltk_data_path = "/tmp/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download NLTK resources on driver node
if not os.path.exists(os.path.join(nltk_data_path, "corpora/stopwords")):
    nltk.download('stopwords', download_dir=nltk_data_path)

english_stopwords = set(stopwords.words('english'))

# Broadcast stopwords to all workers
broadcast_stopwords = spark.sparkContext.broadcast(english_stopwords)

# Define stemming rules for distribution
def stem_word(word):
    """Simple stemming function that follows Snowball stemmer rules"""
    if len(word) < 4:
        return word
        
    if word.endswith('ing'):
        return word[:-3]
    elif word.endswith('ed'):
        return word[:-2]
    elif word.endswith('ies'):
        return word[:-3] + 'y'
    elif word.endswith('es'):
        return word[:-2]
    elif word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    elif word.endswith('ly'):
        return word[:-2]
    
    return word

# Character-by-character tokenizer that uses broadcasted resources
def tokenize_and_normalize(text, docid=-1):
    # Use broadcast stopwords
    stop_words = broadcast_stopwords.value
    tokens = []
    token = ""

    def process_token():
        nonlocal token
        if token:
            # Remove single quotes at start/end
            if token.startswith("'"):
                token = token[1:]
            if token.endswith("'"):
                token = token[:-1]

            # Check if it's a stopword and apply stemming
            if token and token not in stop_words:
                tokens.append(stem_word(token))
            
            token = ""

    # Character-by-character tokenization
    for c in text:
        if c.isalnum() or c == "'":
            token += c.lower()
        else:
            process_token()
    
    process_token()  # Process the last token

    if docid == 1:
        print("\n=== DOC 1 TOKEN DEBUG ===")
        print("Original:", text)
        print("Tokens:   " + "|".join(tokens))
        print("=========================\n")

    return tokens

# Rest of your code remains the same
# Đọc file CSV bằng pandas
file_path = r"/app/wiki_movie_plots_deduped.csv"
df = pd.read_csv(file_path)
# Lấy 30000 dòng đầu tiên để xử lý
small_df = df.head(30000)

# Lấy cột 'Plot' và thêm ID cho mỗi tài liệu
documents = [(i + 1, row['Plot']) for i, row in small_df.iterrows()]

# # Add partition sizing similar to your sum_reduce.py
# def get_optimal_partitions(docs, spark_context):
#     default_parallelism = spark_context.defaultParallelism
#     target_doc_per_partition = 500  # Adjust based on avg doc size
#     size_based = max(default_parallelism, len(docs) // target_doc_per_partition)
#     return min(size_based, default_parallelism * 4)  # Cap at 8x cores

# Tạo RDD từ danh sách các tài liệu
# Tạo RDD từ danh sách các tài liệu với số partition được chỉ định
rdd_start_time = time.perf_counter()
# num_partitions = default_parallelism * 4  # 2× your default parallelism
# num_partitions = get_optimal_partitions(documents, spark.sparkContext)
# rdd = spark.sparkContext.parallelize(documents, num_partitions)
rdd = spark.sparkContext.parallelize(documents)  # Để Spark tự quyết định số partition
rdd_end_time = time.perf_counter()
rdd_time = (rdd_end_time - rdd_start_time) * 1000  # Chuyển sang ms

# Bước 1: Tokenize dùng Spark (phân tán)
tokenize_start_time = time.perf_counter()
tokenized_rdd = rdd.map(lambda doc: (doc[0], tokenize_and_normalize(doc[1], doc[0])))
tokenize_end_time = time.perf_counter()
tokenize_time = (tokenize_end_time - tokenize_start_time) * 1000

# Bước 2 & 3: Tạo word-doc pairs và xây dựng inverted index với Spark
processing_start_time = time.perf_counter()
# Phát ra ((word, doc_id), 1) cho mỗi lần xuất hiện từ
word_doc_ones_rdd = tokenized_rdd.flatMap(
    lambda x: [((word, x[0]), 1) for word in x[1]]
)
# Tính tổng tần suất với reduceByKey
word_doc_freq_rdd = word_doc_ones_rdd.reduceByKey(lambda a, b: a + b)
# Cấu trúc lại thành (word, {doc_id: freq, ...})
grouped_rdd = word_doc_freq_rdd.map(
    lambda x: (x[0][0], (x[0][1], x[1]))
).groupByKey()
# Xây dựng cấu trúc inverted index cuối cùng
inverted_index_rdd = grouped_rdd.mapValues(
    lambda doc_freqs: {doc_id: freq for doc_id, freq in doc_freqs}
)
# Thu thập kết quả
inverted_index = dict(inverted_index_rdd.collect())
processing_end_time = time.perf_counter()
processing_time = processing_end_time - processing_start_time

# Tính số từ trong mỗi tài liệu dùng Spark
doc_word_count_rdd = tokenized_rdd.mapValues(len)
document_word_counts = dict(doc_word_count_rdd.collect())

# Tính tổng số từ đã xử lý
total_words = sum(document_word_counts.values())

# Tính toán số từ xử lý mỗi giây
words_per_second = total_words / ((processing_time) / 1000)

# Hàm tìm kiếm và in kết quả
def search_and_print(inverted_index, keyword, top_n=30):
    start_time = time.perf_counter()
    
    # Chuẩn hóa từ khóa
    normalized_keyword = tokenize_and_normalize(keyword)
    if not normalized_keyword:
        print(f"No results for empty keyword.")
        return
    
    search_word = normalized_keyword[0]
    
    # Lấy dictionary chứa {doc_id: frequency} cho từ cần tìm
    doc_freq = inverted_index.get(search_word, {})
    
    # Sắp xếp kết quả theo tần suất giảm dần
    sorted_results = sorted(doc_freq.items(), 
                           key=lambda x: (-x[1], x[0]))[:top_n]
    
    search_time = (time.perf_counter() - start_time) * 1000
    
    # In báo cáo
    print("\n" + "="*34)
    print("Performance Report:")
    print(f"1. Documents processed:    {len(documents)}")
    print(f"2. Total tokens:           {total_words}")
    print(f"3. Search results for '{keyword}': {len(sorted_results)} docs")
    print(f"4. Time breakdown:")
    print(f"   - Data loading:         {rdd_time:.4f} ms")
    print(f"   - CPU Processing Time: {processing_time *1000:.4f} ms")
    print(f"   - Search operation:     {search_time:.4f} ms")
    # print(f"5. Total execution time:   {total_time:.4f} ms")
    print(f"6. Throughput:             {words_per_second:.4f} tokens/sec")
    print("="*34)
    
    print(f"\nTop {top_n} Results for '{keyword}':")
    for rank, (doc_id, freq) in enumerate(sorted_results, 1):
        print(f"[{rank}] Doc {doc_id} (Freq: {freq})")

# Gọi hàm tìm kiếm
search_keyword = "beer"
search_and_print(inverted_index, search_keyword)

# Dừng SparkSession
spark.stop()