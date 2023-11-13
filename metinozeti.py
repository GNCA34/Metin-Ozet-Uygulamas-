##Gerekli kütüphaneleri import ediyorum.
import tkinter as tk
from tkinter import filedialog
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import string
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from nltk.stem import SnowballStemmer
import re
import string
from nltk import download
from nltk.corpus import stopwords
import gensim.downloader as api
model = api.load("glove-wiki-gigaword-100")
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
nlp = spacy.load('en_core_web_sm')
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.stem import PorterStemmer


#Özel isim kontrolü gerçekleştiriliyor.
def count_proper_nouns(sentence):
    doc = nlp(sentence)
    proper_nouns = [token for token in doc if token.pos_ == 'PROPN']
    count = len(proper_nouns)
    length = len(doc)
    return count/length
def calculate_p1_for_all_sentences():
    global sentences
    p1_values = []
    for i, sentence in enumerate(sentences):
        p1 = count_proper_nouns(sentence)
        print(f"Sentence {i+1} - P1 value: {p1:.2f}")
        p1_values.append(p1)  # P1 değerini diziye ekle
    print(p1_values)
    return p1_values

"""özel isim araştırması 
def count_proper_nouns(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    named_entities = ne_chunk(tagged, binary=True)
    proper_nouns = []
    for subtree in named_entities.subtrees():
        if subtree.label() == 'NE':
            entity = ""
            for leaf in subtree.leaves():
                entity = entity + " " + leaf[0]
            proper_nouns.append(entity.strip())
    count = len(proper_nouns)
    length = len(tokens)
    return count/length


def calculate_p1_for_all_sentences():
    global sentences
    
    for i, sentence in enumerate(sentences):
        p1 = count_proper_nouns(sentence)
        print(f"Sentence {i+1} - P1 value: {p1:.2f}")

"""

#Numerik ifadeleri kontrol ediyorum.
def calculate_p2_for_all_sentences():
    global sentences
    p2_values=[]
    for i, sentence in enumerate(sentences):
        pattern = re.compile(r'\b\d+(?:s|th)?\b')
        matches = pattern.findall(sentence)
        tokens = word_tokenize(sentence.lower())
        words = [word for word in tokens if word not in string.punctuation]
        p2 = len(matches) / len(words)
        p2_values.append(p2)  # p2_values değerini daha sonra skor hesabında kullanacağım
        print(f"Sentence {i+1} - P2 value: {p2:.2f}")      
    print(p2_values)    
    return p2_values       
def calculate_p3_for_all_nodes():
    global similarity_matrix
    p3_values=[]
    threshold = threshold_scale.get()
    num_nodes = len(similarity_matrix)
    
    for i in range(num_nodes):
        p3 = 0
        total_connections = 0

        for j in range(num_nodes):
            if i != j and similarity_matrix[i][j] > threshold:
                p3 += 1

            if similarity_matrix[i][j] > 0:
                total_connections += 1

        p3_value = p3 / total_connections if total_connections > 0 else 0
        p3_values.append(p3_value)
        
        print(f"Node {i+1} - P3 value: {p3_value:.2f}")
    print(p3_values)    
    return p3_values     

def calculate_p4_for_all_sentences():
    global sentences, title
    p4_values=[]  
    # Başlıkta geçen kelimeleri ayır
    title_words = title.split()

    # Kök analizi için PorterStemmer kullan
    stemmer = PorterStemmer()
    
    for i, sentence in enumerate(sentences):
        # Cümleyi küçük harfe dönüştür ve noktalama işaretlerini çıkar
        sentence = sentence.lower().translate(str.maketrans("", "", string.punctuation))
        
        # Cümleyi kelimelere ayır ve köklerini bul
        words = word_tokenize(sentence)
        stemmed_words = [stemmer.stem(word) for word in words]
        
        # Başlıkta geçen kelimelerin köklerini bul
        title_stemmed_words = [stemmer.stem(word) for word in title_words]
        
        # Başlıkta geçen kelimelerin köklerini cümlede say
        count = sum(stemmed_word in title_stemmed_words for stemmed_word in stemmed_words)
        
        # Cümlenin uzunluğunu hesapla
        length = len(words)
        
        # P4 değerini hesapla
        p4 = count / length if length > 0 else 0
        p4_values.append(p4)
        print(f"Sentence {i+1} - P4 value: {p4:.2f}")
    print(p4_values)    
    return p4_values     


def calculate_p5_for_all_sentences():
    global sentences, theme_words
    p5_values = []  # P5 değerlerini saklamak için boş bir dizi oluştur
    
    
    for i, sentence in enumerate(sentences):
        # Cümlenin içinde geçen tema kelime sayısını bul
        theme_word_count = 0
        
        for word, score in theme_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            matches = re.findall(pattern, sentence.lower())
            theme_word_count += len(matches)
        
        # Cümlenin uzunluğunu bul
        sentence_length = len(sentence.split())
        
        # P5 değerini hesapla
        p5 = theme_word_count / sentence_length
        
        p5_values.append(p5)  # P5 değerini diziye ekle
        
        print(f"Sentence {i+1} - P5 value: {p5:.2f}")
    print(p5_values)   
    return p5_values

    
def tokenize_and_stem():
    global sentences, embeddings_list, similarity_matrix, graph
    
    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words("english"))
    
    preprocessed_sentences = []  # Ön işleme yapılan cümlelerin listesi
    embeddings_list = []
    
    for i, sentence in enumerate(sentences):
        
        sentence = sentence.translate(str.maketrans("", "", string.punctuation + "’"))#Noktalama işaretlerini kaldırıyorum
        
        # Tokenize
        tokens = word_tokenize(sentence.lower())
        
        # Stemming
        stemmed_words = [stemmer.stem(word) for word in tokens]
        
        # Stop word elimination
        words = [word for word in stemmed_words if word not in stop_words]
        print(f"Sentence {i+1} - Preprocessed Words: {words}")
        
        # Word embedding
        embeddings = [model[word] for word in words if word in model]
        
        # Calculate sentence embedding vector
        if embeddings:
            sentence_embedding = sum(embeddings) / len(embeddings)
            embeddings_list.append(sentence_embedding)
            preprocessed_sentences.append(words)  # Ön işleme yapılan kelimeleri listeye ekle
            print(f"Sentence {i+1} - Sentence Embedding: {sentence_embedding}")
            
        else:
            print(f"Sentence {i+1} - No embeddings found for the sentence.")
    
    if len(embeddings_list) < 2:
        print("At least 2 sentences are required to calculate similarity.")
    else:
        # Calculate cosine similarity between sentence embeddings
        similarities = cosine_similarity(embeddings_list)
        print("Similarities:")
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                print(f"Similarity between Sentence {i+1} and Sentence {j+1}: {similarities[i][j]}")
    
    
    similarity_matrix = cosine_similarity(embeddings_list)
    calculate_p3_for_all_nodes()
    
    
    # Create graph
    graph = nx.Graph()
    for i, sentence in enumerate(sentences):
        graph.add_node(i, sentence=sentence)
    
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            similarity = similarity_matrix[i][j]
            graph.add_edge(i, j, similarity=similarity)
            
            
    draw_graph(graph)
    print(preprocessed_sentences)

    return preprocessed_sentences  # Ön işleme yapılan kelimelerin listesini döndür

def calculate_tfidf():
    global sentences, theme_words
    
    document = ' '.join(sentences)   #  cümlelerin hepsini kullanarak bu cümlelerden bir belge oluşturuyorum.
    tokens = word_tokenize(document.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    # TF-IDF vektörlerini hesapla
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([document])
    
    # TF-IDF değerlerini al
    tfidf_scores = tfidf_matrix.toarray()[0]
    #dokümandaki toplam kelime sayısının yüzde 10'u tema kelimeler oluyor
    # TF-IDF değeri en yüksek olan kelimeleri temel kelimeler olarak belirle
    theme_words = [(word, score) for score, word in sorted(zip(tfidf_scores, words), reverse=True)][:int(len(words) * 0.1)]
    
    print("Theme Words:")
    for word, score in theme_words:
        print(word, ":", score)
    
    return theme_words


##Burada her cümlenin parametre değerlerine göre cümlelerin skorlarını hesaplıyorum ortalama alarak.
def calculate_values_for_all_sentences():
    global sentences

    p1_values = calculate_p1_for_all_sentences()
    p2_values = calculate_p2_for_all_sentences()
    p3_values = calculate_p3_for_all_nodes()
    p4_values = calculate_p4_for_all_sentences()
    p5_values = calculate_p5_for_all_sentences()
    result = []

    for i in range(len(p1_values)):
        value = (p1_values[i] + p2_values[i] + p3_values[i] + p4_values[i] + p5_values[i]) / 5
        result.append((i, value))  # Her cümlenin indisini ve skorunu birlikte saklayın

    result.sort(key=lambda x: x[1], reverse=True)##Her cümlenin skoruna göre sıralama işlemi yapıyorum.
    
    for i, (sentence_index, value) in enumerate(result):
        print(f"Sentence {sentence_index + 1} - Skor: {value:.2f} - Sıra: {i + 1}")
    top_4_sentences = result[:5]  # Nen burada özet için 4 cümleye kadar aldım

    for i, (sentence_index, value) in enumerate(top_4_sentences):
        sentence = sentences[sentence_index]   
        print(f"Sentence {sentence_index + 1} - Skor: {value:.2f} - Sıra: {i + 1} -  {sentence}")
        
    return top_4_sentences


    
#Burada kendi özetimle gerçek özet arasında ROUGE SKOR hesabı yapıyorum
def calculate_rouge_scores(reference, summary):
    # ROUGE skorlarını hesaplamak için scorer nesnesi oluşturun
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    # ROUGE skorlarını hesaplayın
    scores = scorer.score(reference, summary)

    # Sonuçları döndürün
    return scores
def read_reference_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reference_text = file.read()
    return reference_text

def show_top_sentences():
    top_sentences = calculate_values_for_all_sentences()

    window = tk.Tk()

    # Başlık etiketi ekle
    title_label = tk.Label(window, text="ÖZET")
    title_label.pack()

    
    summary_sentences = []

    # Cümleleri göstermek için etiketler oluşturuluyor.
    for i, (sentence_index, value) in enumerate(top_sentences):
        sentence = sentences[sentence_index]
        summary_sentences.append(sentence)  # Özet cümlelerini listeye ekle

        sentence_label = tk.Label(window, text=f"{i + 1}. Cümle: {sentence}")
        sentence_label.pack()

    # Özet cümlelerini birleştirerek metin formatına dönüştür
    summary_text = '\n'.join(summary_sentences)
    # Referans metni göstermek için bir etiket oluştur
    reference_label = tk.Label(window, text="Gerçek Özet:")
    reference_label.pack()
    
    
    frame = tk.Frame(window, width=400, height=300, bd=1, relief=tk.SOLID)
    frame.pack()

    
    with open("reference.txt", "r", encoding="utf-8") as file:
        reference_text = file.read()
        reference_label = tk.Label(frame, text=reference_text)
        reference_label.pack()


    # ROUGE skorlarını burada hesaplıyorum.
    rouge_scores = calculate_rouge_scores(reference_text, summary_text)

    # ROUGE skorlarını göstermek için etiketler oluşturuluyor.
    rouge1_label = tk.Label(window, text=f"ROUGE-1 Skor Sonucu: {rouge_scores['rouge1'].fmeasure}")
    rouge1_label.pack()


    # Pencereyi göster
    window.mainloop()


root = tk.Tk()
root.title("Sentence Similarity Tool")

# Global Variables
filename = ""
sentences = []
title=[]
theme_words = []
def select_file():
    global filename, sentences, title, theme_words
    
    filename = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("Text files", ".txt"), ("all files", ".*")))
    input_label.config(text="Selected File: " + filename)
    
    # Dosyayı oku ve her satırı bir cümle olarak al
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
        title = lines[0]  # İlk satırı başlık olarak ayarla
        sentences = lines[1:]  # Diğer satırları cümleler olarak al
    
    sentence_text.delete('1.0', tk.END)
    sentence_text.insert('1.0', "\n".join(sentences))
    
    # Dosyadaki cümleleri graf yapısına ekleyip çizdir
    g = process_file()
    draw_graph(g)
    
    # P1 değerlerini hesapla ve yazdır
    calculate_p1_for_all_sentences()
    
    calculate_p2_for_all_sentences()
    
    ##tokenization()
    tokenize_and_stem()

     # Tema kelimelerini hesapla ve yazdır
    calculate_tfidf()
    
    calculate_p5_for_all_sentences()
    calculate_values_for_all_sentences()
    # Arayüzü göster
    show_top_sentences()
def process_file():
    global sentences,title
    
    # Boşluklardan arındır
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Boş graf yapısı oluştur
    g = nx.Graph()
    
    # Her cümleyi graf yapısına bir düğüm olarak ekle
    for i, sentence in enumerate(sentences,start=1):
        g.add_node(i, sentence=sentence)
    
    
    # P4 değerlerini hesapla ve yazdır
    calculate_p4_for_all_sentences()
    
    return g
          
   


def draw_graph(_g):
    fig = plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(_g)
    nx.draw(_g, pos, with_labels=True, node_size=1200, font_size=14,node_color='magenta')
    node_labels = nx.get_node_attributes(_g, 'sentence')
    nx.draw_networkx_labels(_g, pos, labels=node_labels, font_size=6)

    # Cümleler arasında bağlantı çizgilerini çiz
    nx.draw_networkx_edges(_g, pos)

    edge_labels = nx.get_edge_attributes(_g, 'similarity')

    for edge, similarity in edge_labels.items():
        source_node = edge[0]
        target_node = edge[1]
        print(f"Similarity between Sentence {source_node} and Sentence {target_node}: {similarity}")

        # Bağlantı çizgilerine benzerlik değerlerini ekleyin
        x = (pos[source_node][0] + pos[target_node][0]) / 2
        y = (pos[source_node][1] + pos[target_node][1]) / 2
        plt.text(x, y, f"{similarity:.2f}", fontsize=8, horizontalalignment='center', verticalalignment='center',color='blue')
        
    plt.axis('off')

    # Grafik nesnesini Tkinter arayüzüne entegre etmek için FigureCanvasTkAgg kullanıyoruz
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Tkinter arayüzünde grafiği göstermek için plt.show() kullanıyoruz
    plt.show()
    
def set_threshold(threshold_type):
    threshold = threshold_scale.get()
    print(threshold_type, "threshold set to:", threshold)

# Frames
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

graph_frame = tk.Frame(root)
graph_frame.pack(pady=10)

threshold_frame = tk.Frame(root)
threshold_frame.pack(pady=10)

# Input Frame Widgets
select_file_button = tk.Button(input_frame, text="Select File", command=select_file)
select_file_button.grid(row=0, column=0, padx=10)
""""""
input_label = tk.Label(input_frame, text="Selected File: None")
input_label.grid(row=0, column=1, padx=10)

sentence_text = tk.Text(input_frame, height=10, width=50)
sentence_text.grid(row=1, columnspan=2, pady=10)
 
#Graph Frame Widgets           
graph_label = tk.Label(graph_frame, text="Graph will be displayed here.")
graph_label.pack()

# Threshold Frame Widgets
threshold_label = tk.Label(threshold_frame, text="Sentence Similarity Threshold:")
threshold_label.grid(row=0, column=0, padx=10)

threshold_scale = tk.Scale(threshold_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, length=200)
threshold_scale.set(0.5)
threshold_scale.grid(row=0, column=1, padx=10)


cosine_button = tk.Button(threshold_frame, text="Set  Threshold", command=lambda: set_threshold("Cosine"))
cosine_button.grid(row=1, column=1, pady=10)


root.mainloop()