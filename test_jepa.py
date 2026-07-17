"""
🧪 Тестирование обученной Text-JEPA модели
7 сложных задач для проверки качества эмбеддингов
+ Автозагрузка весов и поддержка GPU/CPU
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from itertools import permutations

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import TextJEPAEncoder
from config import Config, DEVICE, NUM_CORES

# ==========================================
# НАСТРОЙКИ
# ==========================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==========================================
# АВТОЗАГРУЗКА И ПОИСК ЧЕКПОИНТОВ
# ==========================================
def find_latest_checkpoint():
    """Находит последний чекпоинт."""
    ckpt_dir = Path(Config.checkpoint_dir)
    if ckpt_dir.exists():
        checkpoints = sorted(ckpt_dir.glob("jepa_shard_*.pt"))
        if checkpoints:
            return str(checkpoints[-1])
    return None


# ==========================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ==========================================
def load_models():
    print("🔄 Загрузка моделей...")

    # Загрузка базовой embedding модели (SentenceTransformers, как в обучении)
    print(f"📦 Загрузка embedding модели: {Config.model_path}")
    st_model = SentenceTransformer(Config.model_path, device=DEVICE.type, trust_remote_code=True)
    tokenizer = st_model.tokenizer
    transformer = st_model[0].auto_model
    max_len = getattr(st_model, 'max_seq_length', Config.max_seq_len)

    config = Config()
    encoder = TextJEPAEncoder(
        config.input_dim, config.hidden_dim, config.embed_dim,
        config.num_layers, config.nhead, config.max_seq_len,
        dropout=config.dropout
    ).to(DEVICE)

    checkpoint_path = find_latest_checkpoint()
    if not checkpoint_path:
        print("❌ ОШИБКА: Чекпоинты не найдены!")
        print("   Сначала запустите обучение: python train_streaming.py ...")
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ckpt['context_encoder_state'])
    encoder.eval()

    step_info = ckpt.get('global_step', ckpt.get('step', 'N/A'))
    print(f"✅ Embedding модель: {Config.model_path}")
    print(f"✅ JEPA чекпоинт: {Path(checkpoint_path).name} (шаг {step_info})")
    print(f"📐 Размерность эмбеддингов: {config.embed_dim}")
    print(f"💻 Устройство: {DEVICE}\n")

    return tokenizer, transformer, max_len, encoder, config


def get_embeddings(texts, tokenizer, transformer, max_len, encoder, config):
    """Получает JEPA-эмбеддинги для списка текстов."""
    # Токенизируем
    inputs = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Получаем эмбеддинги токенов
    with torch.no_grad():
        outputs = transformer(**inputs)
        x = outputs.last_hidden_state.float()

    # Маска паддинга
    key_padding_mask = (inputs["attention_mask"] == 0)

    # JEPA encoder
    with torch.no_grad():
        _, pooled_repr = encoder(x, key_padding_mask)
        pooled_repr = F.normalize(pooled_repr, p=2, dim=-1)

    return pooled_repr.cpu().numpy()


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def print_result(task_name, passed, total, details=""):
    status = "✅ PASS" if passed == total else "⚠️  PARTIAL" if passed > 0 else "❌ FAIL"
    pct = (passed / total * 100) if total > 0 else 0
    print(f"\n{status} | {task_name}: {passed}/{total} ({pct:.0f}%)")
    if details:
        print(f"   💡 {details}")


# ==========================================
# ТЕСТЫ
# ==========================================
def test_paraphrases(encoder, tokenizer, transformer, max_len, config):
    print("=" * 70)
    print("🧪 ТЕСТ 1: ПАРАФРАЗЫ (Paraphrase Detection)")
    print("=" * 70)
    paraphrase_pairs = [
        ("The cat sat on the mat", "A feline was resting upon the rug"),
        ("Machine learning is transforming industries", "Artificial intelligence is revolutionizing business sectors"),
        ("I'm extremely hungry right now", "I could really use something to eat"),
        ("The project deadline has been extended", "We have more time to finish the project"),
        ("That movie was totally awesome!", "The film was exceptionally impressive"),
        ("I gotta go now", "I must depart immediately"),
        ("The dog chased the cat", "The cat was pursued by the dog"),
        ("Scientists discovered a new species", "A new species was found by researchers"),
    ]
    passed = 0
    threshold = Config.paraphrase_threshold
    for sent_a, sent_b in paraphrase_pairs:
        embs = get_embeddings([sent_a, sent_b], tokenizer, transformer, max_len, encoder, config)
        sim = cosine_sim(embs[0], embs[1])
        status = "✓" if sim >= threshold else "✗"
        if sim >= threshold: passed += 1
        print(f"  {status} sim={sim:.3f} | A: {sent_a[:40]}... | B: {sent_b[:40]}...")
    print_result("Парафразы", passed, len(paraphrase_pairs), f"Порог схожести: {threshold}")
    return passed, len(paraphrase_pairs)


def test_odd_one_out(encoder, tokenizer, transformer, max_len, config):
    print("\n" + "=" * 70)
    print("🧪 ТЕСТ 2: БЕЛАЯ ВОРОНА (Odd-one-out)")
    print("=" * 70)
    test_cases = [
        {"theme": "Фрукты", "texts": ["Apples are rich in fiber", "Bananas contain potassium", "Oranges are an excellent source of vitamin C", "Strawberries are sweet red berries", "Grapes can be eaten fresh", "The stock market crashed yesterday"], "odd_idx": 5},
        {"theme": "Программирование", "texts": ["Python is great for data science", "JavaScript runs in web browsers", "Rust provides memory safety", "Go has excellent concurrency support", "TypeScript adds static typing", "My cat loves to sleep in the sun"], "odd_idx": 5},
        {"theme": "Погода", "texts": ["Heavy rain caused flooding", "Snow covered the mountains overnight", "The hurricane made landfall at dawn", "Thunderstorms are expected this afternoon", "A cold front is moving from the north", "The restaurant serves excellent pasta"], "odd_idx": 5},
    ]
    passed = 0
    for case in test_cases:
        print(f"\n  🎯 Тема: {case['theme']}")
        embs = get_embeddings(case["texts"], tokenizer, transformer, max_len, encoder, config)
        avg_sims = []
        for i in range(len(embs)):
            other_sims = [cosine_sim(embs[i], embs[j]) for j in range(len(embs)) if i != j]
            avg_sims.append(np.mean(other_sims))
        predicted_odd = np.argmin(avg_sims)
        is_correct = predicted_odd == case["odd_idx"]
        if is_correct: passed += 1
        status = "✓" if is_correct else "✗"
        print(f"     {status} Предсказан чужак: #{predicted_odd} (avg_sim={avg_sims[predicted_odd]:.3f}) | Настоящий: #{case['odd_idx']}")
    print_result("Белая ворона", passed, len(test_cases), "Чужак = текст с наименьшей средней схожестью")
    return passed, len(test_cases)


def test_noise_robustness(encoder, tokenizer, transformer, max_len, config):
    print("\n" + "=" * 70)
    print("🧪 ТЕСТ 3: УСТОЙЧИВОСТЬ К ШУМУ (Noise Robustness)")
    print("=" * 70)
    original = "Artificial intelligence will transform healthcare in the next decade"
    noisy_versions = [
        ("Опечатки", "Artifical inteligence will transfom healthcare in the next decade"),
        ("Удаление слов", "Artificial intelligence transform healthcare next decade"),
        ("Синонимы", "Machine learning will revolutionize medicine in the coming years"),
        ("Разный регистр", "ARTIFICIAL INTELLIGENCE WILL TRANSFORM HEALTHCARE"),
        ("Перестановка", "Healthcare will be transformed by artificial intelligence"),
        ("Лишние слова", "Well, I think artificial intelligence will definitely transform healthcare"),
    ]
    all_texts = [original] + [v[1] for v in noisy_versions]
    embs = get_embeddings(all_texts, tokenizer, transformer, max_len, encoder, config)
    passed = 0
    threshold = Config.noise_robustness_threshold
    for i, (noise_type, text) in enumerate(noisy_versions):
        sim = cosine_sim(embs[0], embs[i + 1])
        status = "✓" if sim >= threshold else "✗"
        if sim >= threshold: passed += 1
        print(f"  {status} [{noise_type:15s}] sim={sim:.3f}")
    print_result("Устойчивость к шуму", passed, len(noisy_versions), f"Порог: {threshold}")
    return passed, len(noisy_versions)


def test_asymmetric_retrieval(encoder, tokenizer, transformer, max_len, config):
    print("\n" + "=" * 70)
    print("🧪 ТЕСТ 4: АСИММЕТРИЧНЫЙ ПОИСК (Asymmetric Retrieval)")
    print("=" * 70)
    test_cases = [
        {"query": "How to make coffee?", "documents": ["Coffee is prepared by brewing ground coffee beans with hot water. Different methods include espresso, French press, and drip brewing.", "Tea is made by steeping tea leaves in hot water for several minutes. Popular varieties include green, black, and herbal teas.", "The stock market experienced significant volatility today as investors reacted to new economic data released by the Federal Reserve."], "relevant_idx": 0},
        {"query": "Python programming tutorial", "documents": ["The history of ancient Rome spans over a thousand years, from its founding in 753 BC to the fall of the Western Roman Empire.", "Python is a high-level programming language known for its simple syntax. Beginners can start with variables, loops, and functions.", "Mountain climbing requires extensive preparation, proper equipment, and physical conditioning to safely reach the summit."], "relevant_idx": 1},
        {"query": "climate change effects", "documents": ["Basketball is played by two teams of five players on a rectangular court. The objective is to shoot the ball through the opponent's hoop.", "The restaurant serves authentic Italian cuisine with fresh ingredients imported directly from various regions of Italy.", "Rising global temperatures are causing sea levels to rise, extreme weather events to become more frequent, and ecosystems to shift dramatically."], "relevant_idx": 2},
    ]
    passed = 0
    for case in test_cases:
        print(f"\n  🔍 Запрос: '{case['query']}'")
        all_texts = [case["query"]] + case["documents"]
        embs = get_embeddings(all_texts, tokenizer, transformer, max_len, encoder, config)
        query_emb = embs[0]
        doc_embs = embs[1:]
        sims = [cosine_sim(query_emb, doc) for doc in doc_embs]
        predicted_relevant = np.argmax(sims)
        is_correct = predicted_relevant == case["relevant_idx"]
        if is_correct: passed += 1
        status = "✓" if is_correct else "✗"
        print(f"     {status} Найден документ #{predicted_relevant} (sim={sims[predicted_relevant]:.3f})")
    print_result("Асимметричный поиск", passed, len(test_cases), "Query (короткий) vs Documents (длинные)")
    return passed, len(test_cases)


def test_fine_grained_senses(encoder, tokenizer, transformer, max_len, config):
    print("\n" + "=" * 70)
    print("🧪 ТЕСТ 5: ТОНКИЕ СМЫСЛЫ (Word Sense Disambiguation)")
    print("=" * 70)
    test_pairs = [
        ("I deposited money at the bank", "The bank of the river was flooded", False),
        ("I deposited money at the bank", "She works at a financial institution", True),
        ("The bat flew out of the cave at dusk", "He swung the bat and hit a home run", False),
        ("The bat flew out of the cave at dusk", "A nocturnal mammal with wings", True),
        ("Turn on the light please", "This suitcase is very light", False),
        ("Turn on the light please", "The room was bright and sunny", True),
        ("Children play in the park every afternoon", "The musician will play the violin tonight", False),
        ("Shakespeare wrote many famous plays", "Hamlet is a tragic drama performed on stage", True),
    ]
    passed = 0
    close_thr = Config.fine_grained_close_threshold
    far_thr = Config.fine_grained_far_threshold
    for sent_a, sent_b, should_be_close in test_pairs:
        embs = get_embeddings([sent_a, sent_b], tokenizer, transformer, max_len, encoder, config)
        sim = cosine_sim(embs[0], embs[1])
        is_correct = (sim > close_thr) if should_be_close else (sim < far_thr)
        if is_correct: passed += 1
        expected = "близки" if should_be_close else "далеки"
        status = "✓" if is_correct else "✗"
        print(f"  {status} sim={sim:.3f} (ожидается: {expected})")
    print_result("Тонкие смыслы", passed, len(test_pairs), f"Различение многозначных слов (bank, bat, light). Пороги: close>{close_thr}, far<{far_thr}")
    return passed, len(test_pairs)


def test_unsupervised_clustering(encoder, tokenizer, transformer, max_len, config):
    print("\n" + "=" * 70)
    print("🧪 ТЕСТ 6: КЛАСТЕРИЗАЦИЯ БЕЗ УЧИТЕЛЯ (Unsupervised Clustering)")
    print("=" * 70)
    texts = [
        "NASA launched a new satellite into orbit", "Mars rovers are exploring the red planet", "Black holes have immense gravitational pull", "The Milky Way contains billions of stars", "Astronauts trained for the space mission", "Telescopes observe distant galaxies",
        "The chef prepared a delicious pasta dish", "Fresh herbs enhance the flavor of any meal", "Baking requires precise measurements", "Olive oil is a staple in Mediterranean cooking", "The restaurant serves authentic sushi", "Seasoning meat properly is essential",
        "The soccer team won the championship", "Tennis requires quick reflexes and agility", "Marathon runners train for months", "The basketball game went into overtime", "Swimming is excellent cardiovascular exercise", "The Olympic Games bring nations together",
    ]
    true_labels = ([0] * 6) + ([1] * 6) + ([2] * 6)
    embs = get_embeddings(texts, tokenizer, transformer, max_len, encoder, config)

    def simple_kmeans(X, k=3, max_iter=50):
        n = X.shape[0]
        np.random.seed(42)
        centers = X[np.random.choice(n, k, replace=False)]
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            for i in range(n):
                sims = [cosine_sim(X[i], c) for c in centers]
                labels[i] = np.argmax(sims)
            new_centers = []
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    center = X[mask].mean(axis=0)
                    center = center / (np.linalg.norm(center) + 1e-9)
                    new_centers.append(center)
                else:
                    new_centers.append(centers[j])
            new_centers = np.array(new_centers)
            if np.allclose(centers, new_centers): break
            centers = new_centers
        return labels

    predicted = simple_kmeans(embs, k=3)
    best_acc = 0
    best_perm = None
    for perm in permutations([0, 1, 2]):
        mapped = np.array([perm[p] for p in predicted])
        acc = (mapped == np.array(true_labels)).mean()
        if acc > best_acc:
            best_acc = acc
            best_perm = perm

    print(f"\n  🎯 Истинные темы: Космос / Кулинария / Спорт")
    print(f"  📊 Accuracy кластеризации: {best_acc*100:.1f}%")
    passed = 1 if best_acc >= Config.clustering_accuracy_threshold else 0
    print_result("Кластеризация", passed, 1, f"Точность: {best_acc*100:.1f}% (порог: {Config.clustering_accuracy_threshold})")
    return passed, 1


def test_hierarchical_similarity(encoder, tokenizer, transformer, max_len, config):
    print("\n" + "=" * 70)
    print("🧪 ТЕСТ 7: ИЕРАРХИЯ ПОНЯТИЙ (Hierarchical Similarity)")
    print("=" * 70)
    hierarchies = [
        {"name": "Животные", "abstract": "Living organisms in the animal kingdom", "middle": "Domestic dogs are loyal companions", "specific": "The golden retriever puppy played fetch in the park"},
        {"name": "Технологии", "abstract": "Modern technology shapes our daily lives", "middle": "Smartphones have become essential communication devices", "specific": "The iPhone 15 features a titanium frame and USB-C port"},
        {"name": "Еда", "abstract": "Food provides essential nutrients for the body", "middle": "Italian cuisine is famous for pasta and pizza", "specific": "Spaghetti carbonara uses eggs, pecorino, guanciale, and black pepper"},
    ]
    passed = 0
    for h in hierarchies:
        texts = [h["abstract"], h["middle"], h["specific"]]
        embs = get_embeddings(texts, tokenizer, transformer, max_len, encoder, config)
        sim_am = cosine_sim(embs[0], embs[1])
        sim_ms = cosine_sim(embs[1], embs[2])
        sim_as = cosine_sim(embs[0], embs[2])
        print(f"\n  🎯 Иерархия: {h['name']}")
        print(f"     sim(abstract, middle)   = {sim_am:.3f}")
        print(f"     sim(middle, specific)   = {sim_ms:.3f}")
        print(f"     sim(abstract, specific) = {sim_as:.3f}")
        is_correct = (sim_am > sim_as) and (sim_ms > sim_as)
        if is_correct: passed += 1
    print_result("Иерархия понятий", passed, len(hierarchies), "Средний уровень должен быть мостом")
    return passed, len(hierarchies)


# ==========================================
# ГЛАВНАЯ ФУНКЦИЯ
# ==========================================
def main():
    print("🚀 ТЕСТИРОВАНИЕ ОБУЧЕННОЙ TEXT-JEPA МОДЕЛИ")
    print("=" * 70)
    tokenizer, transformer, max_len, encoder, config = load_models()

    results = []
    results.append(("1. Парафразы", test_paraphrases(encoder, tokenizer, transformer, max_len, config)))
    results.append(("2. Белая ворона", test_odd_one_out(encoder, tokenizer, transformer, max_len, config)))
    results.append(("3. Устойчивость к шуму", test_noise_robustness(encoder, tokenizer, transformer, max_len, config)))
    results.append(("4. Асимметричный поиск", test_asymmetric_retrieval(encoder, tokenizer, transformer, max_len, config)))
    results.append(("5. Тонкие смыслы", test_fine_grained_senses(encoder, tokenizer, transformer, max_len, config)))
    results.append(("6. Кластеризация", test_unsupervised_clustering(encoder, tokenizer, transformer, max_len, config)))
    results.append(("7. Иерархия понятий", test_hierarchical_similarity(encoder, tokenizer, transformer, max_len, config)))

    print("\n\n" + "=" * 70)
    print("📊 ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 70)
    total_passed, total_cases = 0, 0
    print(f"\n{'Тест':<30} {'Пройдено':<15} {'Процент':<10}")
    print("-" * 55)
    for name, (passed, cases) in results:
        pct = (passed / cases * 100) if cases > 0 else 0
        total_passed += passed
        total_cases += cases
        print(f"{name:<30} {passed:>6}/{cases:<8} {pct:>6.0f}%")
    total_pct = (total_passed / total_cases * 100) if total_cases > 0 else 0
    print("-" * 55)
    print(f"{'ВСЕГО':<30} {total_passed:>6}/{total_cases:<8} {total_pct:>6.0f}%")

    print("\n" + "=" * 70)
    if total_pct >= 85: print("🏆 ОТЛИЧНО! Модель демонстрирует выдающееся понимание языка")
    elif total_pct >= 70: print("✅ ХОРОШО! Модель успешно решает большинство задач")
    elif total_pct >= 50: print("⚠️  УДОВЛЕТВОРИТЕЛЬНО. Есть потенциал для улучшения")
    else: print("❌ ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНОЕ ОБУЧЕНИЕ или больше данных")
    print("=" * 70)


if __name__ == "__main__":
    main()

