!pip install newspaper3k
!pip install lxml[html_clean]

import requests
from bs4 import BeautifulSoup

cnbc_url = "https://www.cnbc.com/finance/"
headers_dict = {"User-Agent": "Mozilla/5.0"}

def fetch_cnbc_article_links(target_url=cnbc_url):

    page_response = requests.get(target_url, headers=headers_dict)

    html_soup = BeautifulSoup(page_response.content, "html.parser")


    unique_links = set()

    for link_tag in html_soup.find_all("a", href=True):
        url = link_tag["href"]


        if url.startswith("https://www.cnbc.com/") and "/202" in url:

           unique_links.add(url)

    return list(unique_links)

cnbc_links = fetch_cnbc_article_links()

print(f"Total articles fetched: {len(cnbc_links)} ")

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

session = SparkSession.builder \
    .appName("FinInsight_Pipeline") \
    .master("local[*]") \
    .config("spark.ui.port", "4050") \
    .config("spark.driver.memory", "4G") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "1000M") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.3") \
    .getOrCreate()

article_links_list = cnbc_links

parallelized_urls = session.sparkContext.parallelize(article_links_list)

!pip install newspaper3k
!pip install lxml[html_clean]
from newspaper import Article

def get_article_details(link):
    try:
        article_obj = Article(link)
        article_obj.download()
        article_obj.parse()

        return {
            "url": link,
            "title":  article_obj.title,
            "date": str( article_obj.publish_date),
            "content":  article_obj.text
        }

    except Exception:
        return None

filtered_articles_rdd = parallelized_urls.map(get_article_details).filter(lambda record: record is not None)

from pyspark.sql.types import StructType, StructField, StringType

article_structure= StructType([

    StructField("url", StringType(), True),

    StructField("title", StringType(), True),
        StructField("date", StringType(), True),
    StructField("content", StringType(), True),
])

article_dataframe= session.createDataFrame(filtered_articles_rdd, schema=article_structure)

article_dataframe.show(truncate=100)

article_dataframe.write.mode("overwrite").json("output/realtime_financial_data")

!pip install praw
!pip install asyncpraw

import praw
import warnings
from datetime import datetime


warnings.filterwarnings("ignore", category=UserWarning, module="praw")

def fetch_subreddit_posts(subreddit="wallstreetbets", max_posts=500):
    """
    Connect to Reddit and fetch posts from a given subreddit.
    """


    reddit_api = praw.Reddit(
        client_id='MKu1HwTXIGSooUuXaOrzIQ',
        client_secret='sMSkoJf9JM_XON77U-ppAgpb4A-KiA',
        user_agent='reddit_fetcher:v1.0 (by u/Informal-Muffin5689)'
    )

    fetched_posts = []

    for item in list(reddit_api.subreddit(subreddit).new(limit=max_posts)):
        fetched_posts.append({

            "Title": item.title,
            "URL": item.url,
            "Upvotes": item.score,
            "Comments_Count": item.num_comments,
            "Post_Time": datetime.utcfromtimestamp(item.created_utc).strftime('%Y-%m-%d %H:%M:%S')
        })
    return fetched_posts

import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType

reddit_session = SparkSession.builder.appName("RedditETLPipeline").getOrCreate()

data_dir = "/content/reddit_data_raw"
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)

schema_reddit_posts = StructType([
    StructField("Title", StringType(), True),
    StructField("URL", StringType(), True),
    StructField("Upvotes", IntegerType(), True),
    StructField("Comments_Count", IntegerType(), True),
    StructField("Post_Time", StringType(), True)
])

from pyspark.sql.functions import col

raw_data = fetch_subreddit_posts()


if raw_data:


    reddit_df = reddit_session.createDataFrame(
        [(p["Title"], p["URL"], p["Upvotes"], p["Comments_Count"], p["Post_Time"]) for p in raw_data],
        schema=schema_reddit_posts
    )

    reddit_df = reddit_df.withColumn("Post_Time", col("Post_Time").cast(TimestampType()))


    reddit_df.show(5, truncate=True)

    reddit_df.coalesce(1).write \
        .mode("overwrite") \
        .format("json") \
        .option("compression", "none") \
        .save(data_dir)



    print(f"✔️ Stored {reddit_df.count()} posts at {data_dir}")
else:
    print("⚠️ No Reddit data retrieved.")


reddit_session.stop()

!pip install -q pyspark

!wget -q 'https://drive.google.com/uc?export=download&id=11JzGCYd4PNJgQDAG7YSxZ6zDohLnbf1U' -O 'SEC_filings.csv'

import os
if os.path.exists('SEC_filings.csv'):
    print("SEC_filings.csv has been downloaded successfully.")
else:
    print("SEC_filings.csv has not been downloaded")

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

conf = SparkConf().setAppName('SEC_Filings_Ingestion_Pipeline') \
    .set("spark.driver.memory", "4g") \
    .set("spark.executor.memory", "4g")
sc = SparkContext.getOrCreate(conf=conf)

sqlContext = SparkSession.builder \
    .master("local") \
    .appName("SEC_Filings_Processing") \
    .config('spark.ui.port', '4050') \
    .getOrCreate()

print(f"Spark Version: {sqlContext.sparkContext.version}")
print("SparkSession initialized successfully.")

def ingest_file(file_path, sql_context):
    try:
        df = sql_context.read.option("header", "true") \
            .option("inferSchema", "true") \
            .option("quote", "\"") \
            .option("escape", "\"") \
            .csv(file_path)
        print(f"Successfully ingested {file_path}. Total rows: {df.count()}")
        return df
    except Exception as e:
        print(f"Error ingesting CSV: {str(e)}")
        raise

input_path = "SEC_filings.csv"

df_without_changing = ingest_file(input_path, sqlContext)

print("Sample of raw data:")
df_without_changing.show(truncate=False)
print("Schema of raw data:")
df_without_changing.printSchema()

from pyspark.sql.functions import col, lower, regexp_replace, trim, when, isnull
from pyspark.sql.types import FloatType, IntegerType

def data_cleaning(df):
    try:
        cleaning_df = df.withColumn("Name", lower(trim(col("Name")))) \
            .withColumn("Sector", lower(trim(col("Sector")))) \
            .withColumn("Ticker", lower(trim(col("Ticker"))))

        cleaning_df = cleaning_df.withColumn("Name",
            regexp_replace(col("Name"), "<[^>]+>|[^a-zA-Z0-9\\s]", ""))

        cleaning_df = cleaning_df.withColumn("Sector",
            when(isnull(col("Sector")), "unknown").otherwise(col("Sector"))) \
            .withColumn("Market_Value",
                when(isnull(col("Market Value")), 0.0).otherwise(col("Market Value"))) \
            .withColumn("Weight",
                when(isnull(col("Weight (%)")), 0.0).otherwise(col("Weight (%)")))

        cleaning_df = cleaning_df.dropDuplicates(["Ticker", "Name"])

        cleaning_df = cleaning_df.withColumn("Market_Value", col("Market_Value").cast(FloatType())) \
            .withColumn("Weight", col("Weight").cast(FloatType())) \
            .withColumn("Quantity", col("Quantity").cast(IntegerType())) \
            .withColumn("Price", col("Price").cast(FloatType()))

        print(f"Data cleaned. Total rows after cleaning: {cleaning_df.count()}")
        return cleaning_df
    except Exception as e:
        print(f"Error cleaning data: {str(e)}")
        raise

cleaned_df = data_cleaning(df_without_changing)

print("Sample of cleaned data:")
cleaned_df.show(truncate=False)
print("Schema of cleaned data:")
cleaned_df.printSchema()

from pyspark.sql.types import StringType
from pyspark.sql.functions import col

def metadata_extraction(input_df):
    try:
        extracted_metadata = input_df.select(
            col("Ticker").alias("ticker"),
            col("Name").alias("company_name"),
            col("Sector").alias("sector"),
            col("Market_Value").alias("market_value"),
            col("Weight").alias("weight_percent"),
            col("Quantity").alias("shares_quantity"),
            col("Price").alias("share_price"),
            col("Location").alias("country"),
            col("Exchange").alias("stock_exchange"),
            col("Accrual Date").alias("accrual_date")
        )
        extracted_metadata = extracted_metadata.withColumn("record_id",
            col("ticker").cast(StringType()) + "_" + col("company_name").cast(StringType()))
        print("Metadata extracted successfully.")
        return extracted_metadata
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        raise

extracted_metadata = metadata_extraction(cleaned_df)

print("Sample of metadata:")
extracted_metadata.show(truncate=False)
print("Schema of metadata:")
extracted_metadata.printSchema()

def data_storing(df, output_path):
    try:
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
        print(f"Data successfully saved to {output_path}")

        df_storing = sqlContext.read.option("header", "true").csv(output_path)
        print(f"Verification: Read back {df_storing.count()} rows from {output_path}")
        print("Sample of stored data:")
        df_storing.show(truncate=False)
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        raise

output_path = "processed_sec_filings"

data_storing(extracted_metadata, output_path)

!pip install -q pyspark==3.4.1 spark-nlp==5.1.3
!pip install faiss-cpu

from pyspark.sql import SparkSession

spark_session = SparkSession.getActiveSession()

if spark_session:
    spark_session.stop()

import sparknlp
spark = sparknlp.start()
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from sentence_transformers import SentenceTransformer
from sparknlp.pretrained import ResourceDownloader, PretrainedPipeline
import pandas as pd
import numpy as np
import faiss
import os

df1 = spark.read.json("/content/data/CNBC_financial_news_1.json")
df2 = spark.read.json("/content/data/CNBC_financial_articles_2.json")
cnbc_df = df1.union(df2)

reddit_df = spark.read.json("/content/data/reddit_posts.json")

def clean_text(df, col_name):
    return df.withColumn(col_name, lower(col(col_name))) \
             .withColumn(col_name, regexp_replace(col(col_name), "<.*?>", "")) \
             .withColumn(col_name, regexp_replace(col(col_name), "[^a-zA-Z0-9\\s]", ""))

cnbc_df = clean_text(cnbc_df, "content")
reddit_df = clean_text(reddit_df, "Title")

ner_pipeline = PretrainedPipeline("recognize_entities_dl", lang="en", remote_loc=None)

sample_text = cnbc_df.select("content").limit(1).collect()[0][0]
ner_result = ner_pipeline.fullAnnotate(sample_text)
print(ner_result[0]["entities"])

model = SentenceTransformer("all-MiniLM-L6-v2")


cnbc_pd = cnbc_df.select("title", "content", "date", "url").toPandas()
reddit_pd = reddit_df.select("Title", "URL", "Post_Time").toPandas()
reddit_pd.rename(columns={"Title": "title", "URL": "url", "Post_Time": "date"}, inplace=True)

combined = pd.concat([cnbc_pd, reddit_pd], ignore_index=True)

combined['content'] = combined['content'].fillna('')

combined["text"] = combined["title"] + " " + combined["content"]

combined['text'] = combined['text'].astype(str)

combined["embedding"] = combined["text"].apply(lambda x: model.encode(x).tolist())

dimension = len(combined["embedding"][0])
index = faiss.IndexFlatL2(dimension)

embedding_matrix = np.vstack(combined["embedding"].values)
index.add(embedding_matrix)

def search_similar(query, k=3):
    q_embed = model.encode([query])
    distances, indices = index.search(np.array(q_embed), k)
    return combined.iloc[indices[0]][["title", "text", "url"]]

results = search_similar("What’s the impact of Trump’s tariffs on inflation?")
print(results)

results = search_similar("Apple is acquiring a startup in New York.")
print(results)

import json

file_paths = [
    "/content/data/reddit_posts.json",
    "/content/data/CNBC_financial_news_1.json",
    "/content/data/CNBC_financial_articles_2.json"
]

all_documents = []

for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_documents.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

processed_docs = []
text_fields = ["Title", "Content", "Summary", "Text", "text", "headline", "body"]

for doc in all_documents:
    text = ""
    for field in text_fields:
        if field in doc and isinstance(doc[field], str):
            text += doc[field].strip() + " "
    text = text.strip()
    if text:
        processed_docs.append({"text": text})

model = SentenceTransformer("all-MiniLM-L6-v2")
corpus = [doc["text"] for doc in processed_docs]
embeddings = model.encode(corpus, convert_to_numpy=True).astype("float32")

dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(embeddings)

import pickle
import faiss

faiss.write_index(faiss_index, "faiss_index.bin")

with open("processed_docs.pkl", "wb") as f:
    pickle.dump(processed_docs, f)

!pip install langchain-community

from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": 0.7})
    return llm

def run_rag(query: str, faiss_index, documents, embed_model, k=7):
    query_vec = embed_model.encode(query).astype('float32').reshape(1, -1)

    D, I = faiss_index.search(query_vec, k)
    retrieved = [documents[i]["text"][:1000] for i in I[0] if i < len(documents)]
    context = "\n\n".join(retrieved)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
       template="""
Use the following financial information to answer the user's question in a factual and complete way. Use numbered citations if helpful.

Context:
{context}

Question:
{question}

Answer (with sources if relevant):
"""


    )

    llm = load_llm()
    rag_chain = prompt | llm
    result = rag_chain.invoke({"context": context, "question": query})

    print("📘 Answer:\n", result)
    return result

query = "What are the risks of Amazon’s $15B warehouse expansion?"
run_rag(query, faiss_index, processed_docs, model)

!pip install textstat

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import json

def clean_text(df, col_name):
    return df.withColumn(col_name, lower(col(col_name))) \
             .withColumn(col_name, regexp_replace(col(col_name), "<.*?>", "")) \
             .withColumn(col_name, regexp_replace(col(col_name), "[^a-zA-Z0-9\\s]", ""))

df1 = spark.read.json("/content/data/CNBC_financial_news_1.json")
df2 = spark.read.json("/content/data/CNBC_financial_articles_2.json")
cnbc_df = df1.union(df2)
cnbc_df = clean_text(cnbc_df, "content")
reddit_df = spark.read.json("/content/data/reddit_posts.json")
reddit_df = clean_text(reddit_df, "Title")

cnbc_pd = cnbc_df.select("title", "content", "date", "url").toPandas()
reddit_pd = reddit_df.select("Title", "URL", "Post_Time").toPandas()
reddit_pd.rename(columns={"Title": "title", "URL": "url", "Post_Time": "date"}, inplace=True)
combined = pd.concat([cnbc_pd, reddit_pd], ignore_index=True)
combined['content'] = combined['content'].fillna('')
combined["text"] = combined["title"] + " " + combined["content"]
combined['text'] = combined['text'].astype(str)

model = SentenceTransformer("all-MiniLM-L6-v2")
combined["embedding"] = combined["text"].apply(lambda x: model.encode(x).tolist())
dimension = len(combined["embedding"][0])
index = faiss.IndexFlatL2(dimension)
embedding_matrix = np.vstack(combined["embedding"].values)
index.add(embedding_matrix)

file_paths = ["/content/data/reddit_posts.json", "/content/data/CNBC_financial_articles_2.json", "/content/data/CNBC_financial_news_1.json"]
all_documents = []
for file_path in file_paths:
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    all_documents.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

processed_docs = []
text_fields = ["Title", "Content", "Summary", "Text", "text", "headline", "body"]
for doc in all_documents:
    text = " ".join([doc[field].strip() for field in text_fields if field in doc and isinstance(doc[field], str)]).strip()
    if text:
        processed_docs.append({"text": text})

corpus = [doc["text"] for doc in processed_docs]
embeddings = model.encode(corpus, convert_to_numpy=True).astype("float32")
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(embeddings)

def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
    return HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": 0.7})

def run_rag(query, faiss_index, documents, embed_model, k=7):
    query_vec = embed_model.encode(query).astype('float32').reshape(1, -1)
    D, I = faiss_index.search(query_vec, k)
    retrieved = [documents[i]["text"][:1000] for i in I[0] if i < len(documents)]
    context = "\n\n".join(retrieved)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Use the following financial information to answer the user's question clearly and factually. Cite sources by number. If context lacks specific information, state so.
        Context: {context}
        Question: {question}
        Answer (with numbered citations):
        """
    )
    llm = load_llm()
    rag_chain = prompt | llm
    result = rag_chain.invoke({"context": context, "question": query})
    return result, retrieved

def evaluate_response(query, response, retrieved_docs):
    evaluation = {"query": query, "response": response}
    doc_texts = " ".join(retrieved_docs).lower()
    query_words = query.lower().split()
    relevant_terms = [word for word in query_words if word in doc_texts]
    accuracy_score = len(relevant_terms) / len(query_words)
    evaluation["accuracy"] = "High" if accuracy_score > 0.5 else "Low"
    evaluation["accuracy_score"] = accuracy_score

    word_count = len(response.split())
    sentence_count = response.count(".") + 1
    evaluation["clarity"] = "High" if word_count < 100 and sentence_count > 1 else "Low"
    grounding_score = (0.5 if "[1]" in response else 0) + (0.5 if any(doc.lower()[:50] in response.lower() for doc in retrieved_docs) else 0)
    evaluation["grounding"] = "High" if grounding_score > 0.5 else "Low"
    evaluation["grounding_score"] = grounding_score
    return evaluation

query = "What are the risks of Amazon’s $15B warehouse expansion?"
response, retrieved_docs = run_rag(query, faiss_index, processed_docs, model)
eval_result = evaluate_response(query, response, retrieved_docs)

print("=== Checking Accuracy, Clarity, and Grounding ===")
print(f"Query: {query}")
print(f"Answer:\n{response}")
print(f"Retrieved Documents (Top 3):\n{[doc[:100] + '...' for doc in retrieved_docs[:3]]}")
print(f"Evaluation:")
print(f"  - Accuracy: {eval_result['accuracy']} (Score: {eval_result['accuracy_score']:.2f})")
print(f"  - Clarity: {eval_result['clarity']} (Words: {len(response.split())})")
print(f"  - Grounding: {eval_result['grounding']} (Score: {eval_result['grounding_score']:.2f})")

with open("llm_evaluation.json", "w") as f:
    json.dump(eval_result, f, indent=2)
print("\nEvaluation saved to llm_evaluation.json")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from functools import lru_cache
import time
import json

import sparknlp
spark = sparknlp.start()

def clean_text(df, col_name):
    return df.withColumn(col_name, lower(col(col_name))) \
             .withColumn(col_name, regexp_replace(col(col_name), "<.*?>", "")) \
             .withColumn(col_name, regexp_replace(col(col_name), "[^a-zA-Z0-9\\s]", ""))

df1 = spark.read.json("/content/data/CNBC_financial_news_1.json")
df2 = spark.read.json("/content/data/CNBC_financial_articles_2.json")
cnbc_df = df1.union(df2)
cnbc_df = clean_text(cnbc_df, "content")
reddit_df = spark.read.json("/content/data/reddit_posts.json")
reddit_df = clean_text(reddit_df, "Title")
cnbc_pd = cnbc_df.select("title", "content", "date", "url").toPandas()
reddit_pd = reddit_df.select("Title", "URL", "Post_Time").toPandas()
reddit_pd.rename(columns={"Title": "title", "URL": "url", "Post_Time": "date"}, inplace=True)
combined = pd.concat([cnbc_pd, reddit_pd], ignore_index=True)
combined['content'] = combined['content'].fillna('')
combined["text"] = combined["title"] + " " + combined["content"]
combined['text'] = combined['text'].astype(str)

model = SentenceTransformer("all-MiniLM-L6-v2")
combined["embedding"] = combined["text"].apply(lambda x: model.encode(x).tolist())
dimension = len(combined["embedding"][0])
index = faiss.IndexFlatL2(dimension)
embedding_matrix = np.vstack(combined["embedding"].values)
index.add(embedding_matrix)

file_paths = ["/content/data/reddit_posts.json", "/content/data/CNBC_financial_articles_2.json", "/content/data/CNBC_financial_news_1.json"]
all_documents = []
for file_path in file_paths:
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    all_documents.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
processed_docs = []
text_fields = ["Title", "Content", "Summary", "Text", "text", "headline", "body"]
for doc in all_documents:
    text = " ".join([doc[field].strip() for field in text_fields if field in doc and isinstance(doc[field], str)]).strip()
    if text:
        processed_docs.append({"text": text})
corpus = [doc["text"] for doc in processed_docs]
embeddings = model.encode(corpus, convert_to_numpy=True).astype("float32")
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(embeddings)

@lru_cache(maxsize=1000)
def embed_query(query: str):
    return model.encode(query).astype('float32')

def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
    return HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": 0.7})

def run_rag(query, faiss_index, documents, embed_model, k=7):
    start_time = time.time()
    query_vec = embed_query(query).reshape(1, -1)
    D, I = faiss_index.search(query_vec, k)
    retrieved = [documents[i]["text"][:1000] for i in I[0] if i < len(documents)]
    context = "\n\n".join(retrieved)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Use the following financial information to answer the user's question clearly and factually. Cite sources by number. If context lacks specific information, state so.
        Context: {context}
        Question: {question}
        Answer (with numbered citations):
        """
    )
    llm = load_llm()
    rag_chain = prompt | llm
    result = rag_chain.invoke({"context": context, "question": query})
    latency = time.time() - start_time
    return result, retrieved, latency

print("=== Testing Diverse Financial Queries ===")
queries = [
    "What are the risks of Amazon’s $15B warehouse expansion?",
    "What is the impact of Trump’s tariffs on inflation?",
    "Apple is acquiring a startup in New York."
]
for query in queries:
    response, retrieved_docs, latency = run_rag(query, faiss_index, processed_docs, model)
    print(f"\nQuery: {query}")
    print(f"Answer:\n{response}")
    print(f"Retrieved (Top 3):\n{[doc[:100] + '...' for doc in retrieved_docs[:3]]}")
    print(f"Latency: {latency:.2f}s")

print("\n=== Fine-Tuning Retrieval ===")
query = "What are the risks of Amazon’s $15B warehouse expansion?"
response, retrieved_docs, latency = run_rag(query, faiss_index, processed_docs, model, k=3)
print(f"Query: {query}")
print(f"Answer (k=3):\n{response}")
print(f"Retrieved (Top 3):\n{[doc[:100] + '...' for doc in retrieved_docs[:3]]}")
print(f"Latency: {latency:.2f}s")

print("\n=== Optimize and Finalize ===")
nlist = 100
quantizer = faiss.IndexFlatIP(dimension)
optimized_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
optimized_index.train(embeddings)
optimized_index.add(embeddings)

response, retrieved_docs, latency = run_rag(query, optimized_index, processed_docs, model)
print(f"Query (Optimized): {query}")
print(f"Answer:\n{response}")
print(f"Latency: {latency:.2f}s")