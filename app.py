from flask import Flask, request, render_template, session
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

nltk.download('punkt')

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required to use session
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

chunking_methods = [
    'Sentence-Level Chunking',
    'Fixed-Length Chunking',
    'Sliding Window Chunking',
    'Paragraph-Level Chunking',
    'Keyword-Based Chunking',
    'Semantic-Based Chunking'
]

embedding_methods = [
    'TF-IDF',
    'Sentence Transformers'
]

# Load Sentence Transformers model
sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')


def sentence_level_chunking(text):
    return sent_tokenize(text)


def fixed_length_chunking(text, chunk_size=50):
    words = word_tokenize(text)
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def sliding_window_chunking(text, window_size=50, step=25):
    words = word_tokenize(text)
    return [' '.join(words[i:i + window_size]) for i in range(0, len(words) - window_size + 1, step)]


def paragraph_level_chunking(text):
    return text.split('\n\n')


def keyword_based_chunking(text, keywords=['important', 'note', 'key']):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = []
    for sentence in sentences:
        chunk.append(sentence)
        if any(keyword in sentence.lower() for keyword in keywords):
            chunks.append(' '.join(chunk))
            chunk = []
    if chunk:
        chunks.append(' '.join(chunk))
    return chunks


def semantic_based_chunking(text):
    sentences = sent_tokenize(text)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    clusters = []
    current_cluster = [sentences[0]]
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(tfidf_matrix[i-1:i], tfidf_matrix[i:i+1])
        if similarity[0][0] < 0.5:  # Adjust threshold as needed
            clusters.append(' '.join(current_cluster))
            current_cluster = []
        current_cluster.append(sentences[i])
    if current_cluster:
        clusters.append(' '.join(current_cluster))
    return clusters


def auto_chunk(text):
    # Placeholder: Implement logic to determine the best chunking method dynamically
    return sentence_level_chunking(text)


def calculate_metrics(chunks):
    chunk_lengths = [len(chunk.split()) for chunk in chunks]
    avg_chunk_size = np.mean(chunk_lengths)
    total_chunks = len(chunks)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(chunks)
    pairwise_similarities = cosine_similarity(tfidf_matrix)
    semantic_coherence = np.mean(pairwise_similarities)
    redundancy = np.mean(np.triu(pairwise_similarities, k=1))

    return {
        'Average Chunk Size': avg_chunk_size,
        'Total Number of Chunks': total_chunks,
        'Semantic Coherence': semantic_coherence,
        'Redundancy': redundancy
    }


@app.route('/', methods=['GET', 'POST'])
def index():
    metrics = {}
    ranked_chunks = []
    chunks = []
    selected_chunking_method = None
    selected_embedding_method = None
    text = None
    query = None

    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            with open(filepath, 'r') as f:
                text = f.read()
            session['text'] = text  # Save text in session
        else:
            text = session.get('text')  # Use previously uploaded text

        if not text:
            return "No file uploaded. Please upload a file."

        if 'chunking_button' in request.form:
            selected_chunking_method = request.form['chunking_method']
            if selected_chunking_method == 'Sentence-Level Chunking':
                chunks = sentence_level_chunking(text)
            elif selected_chunking_method == 'Fixed-Length Chunking':
                chunks = fixed_length_chunking(text)
            elif selected_chunking_method == 'Sliding Window Chunking':
                chunks = sliding_window_chunking(text)
            elif selected_chunking_method == 'Paragraph-Level Chunking':
                chunks = paragraph_level_chunking(text)
            elif selected_chunking_method == 'Keyword-Based Chunking':
                chunks = keyword_based_chunking(text)
            elif selected_chunking_method == 'Semantic-Based Chunking':
                chunks = semantic_based_chunking(text)
            elif selected_chunking_method == 'Auto-Chunk':
                chunks = auto_chunk(text)

            metrics = calculate_metrics(chunks)

        if 'retrieval_button' in request.form:
            selected_chunking_method = request.form['chunking_method']
            selected_embedding_method = request.form['embedding_method']
            query = request.form['query']

            if selected_chunking_method == 'Sentence-Level Chunking':
                chunks = sentence_level_chunking(text)
            elif selected_chunking_method == 'Fixed-Length Chunking':
                chunks = fixed_length_chunking(text)
            elif selected_chunking_method == 'Sliding Window Chunking':
                chunks = sliding_window_chunking(text)
            elif selected_chunking_method == 'Paragraph-Level Chunking':
                chunks = paragraph_level_chunking(text)
            elif selected_chunking_method == 'Keyword-Based Chunking':
                chunks = keyword_based_chunking(text)
            elif selected_chunking_method == 'Semantic-Based Chunking':
                chunks = semantic_based_chunking(text)
            elif selected_chunking_method == 'Auto-Chunk':
                chunks = auto_chunk(text)

            if query:
                chunk_embeddings = []
                query_embedding = []

                if selected_embedding_method == 'TF-IDF':
                    tfidf_vectorizer = TfidfVectorizer()
                    chunk_embeddings = tfidf_vectorizer.fit_transform(chunks).toarray()
                    query_embedding = tfidf_vectorizer.transform([query]).toarray()
                elif selected_embedding_method == 'Sentence Transformers':
                    chunk_embeddings = sentence_transformer_model.encode(chunks)
                    query_embedding = sentence_transformer_model.encode([query])

                similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
                ranked_chunks = sorted(zip(chunks, similarities), key=lambda x: x[1], reverse=True)

    return render_template(
        'index.html',
        chunking_methods=chunking_methods + ['Auto-Chunk'],
        embedding_methods=embedding_methods,
        metrics=metrics,
        chunks=chunks,
        ranked_chunks=ranked_chunks,
        selected_chunking_method=selected_chunking_method,
        selected_embedding_method=selected_embedding_method,
        query=query,
        text=session.get('text')
    )


if __name__ == '__main__':
    app.run(debug=True)