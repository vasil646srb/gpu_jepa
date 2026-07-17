"""
🧪 Тестирование обученной Text-JEPA модели
Все эмбеддинги предвычисляются на GPU и хранятся в RAM.
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
from config import Config, DEVICE

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==========================================
# ПРЕДВЫЧИСЛЕНИЕ ЭМБЕДДИНГОВ НА GPU
# ==========================================
class EmbeddingCache:
    """Кэширует эмбеддинги текстов в RAM (GPU или CPU)."""

    def __init__(self, model_path, device='cuda'):
        print(f"📦 Загрузка embedding модели: {model_path}")
        self.device = device
        self.st_model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
        self.tokenizer = self.st_model.tokenizer
        self.transformer = self.st_model[0].auto_model
        self.max_len = getattr(self.st_model, 'max_seq_length', Config.max_seq_len)
        self._cache = {}  # text -> (x_tensor, mask_tensor) on GPU
        print(f"   ✅ Модель загружена на {device}")
        print(f"   📐 max_seq_length: {self.max_len}")

    def _compute(self, texts):
        """Вычисляет эмбеддинги для списка текстов."""
        inputs = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.transformer(**inputs)
            x = outputs.last_hidden_state.float()

        mask = (inputs["attention_mask"] == 0)
        return x, mask

    def get(self, texts):
        """Возвращает эмбеддинги (кэширует новые)."""
        # Находим тексты, которых ещё нет в кэше
        new_texts = [t for t in texts if t not in self._cache]

        if new_texts:
            # Вычисляем батчем для эффективности
            batch_size = 32
            for i in range(0, len(new_texts), batch_size):
                batch = new_texts[i:i + batch_size]
                x, mask = self._compute(batch)
                for j, txt in enumerate(batch):
                    self._cache[txt] = (x[j:j+1], mask[j:j+1])

        # Возвращаем из кэша
        xs = []
        masks = []
        for t in texts:
            x, m = self._cache[t]
            xs.append(x)
            masks.append(m)

        return torch.cat(xs, dim=0), torch.cat(masks, dim=0)

    def get_jepa_embeddings(self, texts, encoder):
        """Получает финальные JEPA-эмбеддинги для текстов."""
        x, mask = self.get(texts)
        with torch.no_grad():
            _, pooled = encoder(x, mask)
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled.cpu().numpy()

    def clear_cache(self):
        """Очищает кэш для освобождения памяти."""
        self._cache.clear()
        torch.cuda.empty_cache()
        print("🗑  Кэш эмбеддингов очищен")


# ==========================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ==========================================
def load_models():
    print("🔄 Загрузка моделей...")

    # Embedding модель на GPU (кэшируем эмбеддинги в RAM)
    cache = EmbeddingCache(Config.model_path, device=DEVICE.type)

    # JEPA encoder на GPU
    config = Config()
    encoder = TextJEPAEncoder(
        config.input_dim, config.hidden_dim, config.embed_dim,
        config.num_layers, config.nhead, config.max_seq_len,
        dropout=config.dropout
    ).to(DEVICE)

    # Чекпоинт
    ckpt_dir = Path(Config.checkpoint_dir)
    ckpts = sorted(ckpt_dir.glob("jepa_shard_*.pt"))
    if not ckpts:
        print("❌ Чекпоинты не найдены!"); sys.exit(1)

    ckpt = torch.load(ckpts[-1], map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ckpt['context_encoder_state'])
    encoder.eval()

    step_info = ckpt.get('global_step', 'N/A')
    print(f"✅ JEPA чекпоинт: {ckpts[-1].name} (шаг {step_info})")
    print(f"📐 Размерность эмбеддингов: {config.embed_dim}")
    print(f"💻 Устройство: {DEVICE}\n")

    return cache, encoder, config


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
def test_paraphrases(cache, encoder, config):
    print("=" * 70)
    print("🧪 ТЕСТ 1: ПАРАФРАЗЫ (Paraphrase Detection)")
    print("=" * 70)
    pairs = [
        ("The cat sat on the mat", "A feline was resting upon the rug"),
        ("Machine learning is transforming industries", "Artificial intelligence is revolutionizing business sectors"),
        ("I'm extremely hungry right now", "I could really use something to eat"),
        ("The project deadline has been extended", "We have more time to finish the project"),
        ("That movie was totally awesome!", "The film was exceptionally impressive"),
        ("I gotta go now", "I must depart immediately"),
        ("The dog chased the cat", "The cat was pursued by the dog"),
        ("Scientists discovered a new species", "A new species was found by researchers"),
    ]
    # Предвычисляем все уникальные тексты
    all_texts = list(set([t for pair in pairs for t in pair]))
    cache.get(all_texts)  # кэшируем

    passed = 0
    for sent_a, sent_b in pairs:
        embs = cache.get_jepa_embeddings([sent_a, sent_b], encoder)
        sim = cosine_sim(embs[0], embs[1])
        status = "✓" if sim >= Config.paraphrase_threshold else "✗"
        if sim >= Config.paraphrase_threshold: passed += 1
        print(f"  {status} sim={sim:.3f} | A: {sent_a[:40]}... | B: {sent_b[:40]}...")
    print_result("Парафразы", passed, len(pairs), f"Порог: {Config.paraphrase_threshold}")
    return passed, len(pairs)


def test_odd_one_out(cache, encoder, config):
    print("\n" + "=" * 70)
    print("🧪 ТЕСТ 2: БЕЛАЯ ВОРОНА (Odd-one-out)")
    print("=" * 70)
    cases = [
        {"theme": "Фрукты", "texts": ["Apples are rich in fiber", "Bananas contain potassium", "Oranges are an excellent source of vitamin C", "Strawberries are sweet red berries", "Grapes can be eaten fresh", "The stock market crashed yesterday"], "odd_idx": 5},
        {"theme": "Программирование", "texts": ["Python is great for data science", "JavaScript runs in web browsers", "Rust provides memory safety", "Go has excellent concurrency support", "TypeScript adds static typing", "My cat loves to sleep in the sun"], "odd_idx": 5},
        {"theme": "Погода", "texts": ["Heavy rain caused flooding", "Snow covered the mountains overnight", "The hurricane made landfall at dawn", "Thunderstorms are expected this afternoon", "A cold front is moving from the north", "The restaurant serves excellent pasta"], "odd_idx": 5},
    ]
    # Кэшируем все тексты
    all_texts = list(set([t for c in cases for t in c["texts"]]))
    cache.get(all_texts)

    passed = 0
    for case in cases:
        print(f"\n  🎯 Тема: {case['theme']}")
        embs = cache.get_jepa_embeddings(case["texts"], encoder)
        avg_sims = []
        for i in range(len(embs)):
            other_sims = [cosine_sim(embs[i], embs[j]) for j in range(len(embs)) if i != j]
            avg_sims.append(np.mean(other_sims))
        predicted_odd = np.argmin(avg_sims)
        is_correct = predicted_odd == case["odd_idx"]
        if is_correct: passed += 1
        status = "✓" if is_correct else "✗"
        print(f"     {status} Чужак: #{predicted_odd} | Настоящий: #{case['odd_idx']}")
    print_result("Белая ворона", passed, len(cases))
    return passed, len(cases)


def test_noise_robustness(cache, encoder, config):
    print("\n" + "=" * 70)
    print("🧪 ТЕСТ 3: УСТОЙЧИВОСТЬ К ШУМУ (Noise Robustness)")
    print("=" * 70)
    original = "Artificial intelligence will transform healthcare in the next decade"
    noisy = [
        ("Опечатки", "Artifical inteligence will transfom healthcare in the next decade"),
        ("Удаление слов", "Artificial intelligence transform healthcare next decade"),
        ("Синонимы", "Machine learning will revolutionize medicine in the coming years"),
        ("Разный регистр", "ARTIFICIAL INTELLIGENCE WILL TRANSFORM HEALTHCARE"),
        ("Перестановка", "Healthcare will be transformed by artificial intelligence"),
        ("Лишние слова", "Well, I think artificial intelligence will definitely transform healthcare"),
    ]
    all_texts = [original] + [t[1] for t in noisy]
    cache.get(all_texts)

    passed = 0
    embs = cache.get_jepa_embeddings(all_texts, encoder)
    for i, (noise_type, text) in enumerate(noisy):
        sim = cosine_sim(embs[0], embs[i + 1])
        status = "✓" if sim >= Config.noise_robustness_threshold else "✗"
        if sim >= Config.noise_robustness_threshold: passed += 1
        print(f"  {status} [{noise_type:15s}] sim={sim:.3f}")
    print_result("Устойчивость к шуму", passed, len(noisy), f"Порог: {Config.noise_robustness_threshold}")
    return passed, len(noisy)


def test_asymmetric_retrieval(cache, encoder, config):
    print("\n" + "=" * 70)
    print("🧪 ТЕСТ 4: АСИММЕТРИЧНЫЙ ПОИСК (Asymmetric Retrieval)")
    print("=" * 70)
    cases = [
        {"query": "How to make coffee?", "docs": ["Coffee is prepared by brewing ground coffee beans with hot water. Different methods include espresso, French press, and drip brewing.", "Tea is made by steeping tea leaves in hot water for several minutes. Popular varieties include green, black, and herbal teas.", "The stock market experienced significant volatility today as investors reacted to new economic data released by the Federal Reserve."], "rel": 0},
        {"query": "Python programming tutorial", "docs": ["The history of ancient Rome spans over a thousand years, from its founding in 753 BC to the fall of the Western Roman Empire.", "Python is a high-level programming language known for its simple syntax. Beginners can start with variables, loops, and functions.", "Mountain climbing requires extensive preparation, proper equipment, and physical conditioning to safely reach the summit."], "rel": 1},
        {"query": "climate change effects", "docs": ["Basketball is played by two teams of five players on a rectangular court. The objective is to shoot the ball through the opponent's hoop.", "The restaurant serves authentic Italian cuisine with fresh ingredients imported directly from various regions of Italy.", "Rising global temperatures are causing sea levels to rise, extreme weather events to become more frequent, and ecosystems to shift dramatically."], "rel": 2},
    ]
    all_texts = [c["query"] for c in cases] + [d for c in cases for d in c["docs"]]
    cache.get(list(set(all_texts)))

    passed = 0
    for case in cases:
        print(f"\n  🔍 Запрос: '{case['query']}'")
        all_embs = cache.get_jepa_embeddings([case["query"]] + case["docs"], encoder)
        query_emb = all_embs[0]
        doc_embs = all_embs[1:]
        sims = [cosine_sim(query_emb, d) for d in doc_embs]
        predicted = np.argmax(sims)
        is_correct = predicted == case["rel"]
        if is_correct: passed += 1
        status = "✓" if is_correct else "✗"
        print(f"     {status} Документ #{predicted} (sim={sims[predicted]:.3f})")
    print_result("Асимметричный поиск", passed, len(cases))
    return passed, len(cases)


def test_fine_grained_senses(cache, encoder, config):
    print("\n" + "=" * 70)
    print("🧪 ТЕСТ 5: ТОНКИЕ СМЫСЛЫ (Word Sense Disambiguation)")
    print("=" * 70)
    pairs = [
        ("I deposited money at the bank", "The bank of the river was flooded", False),
        ("I deposited money at the bank", "She works at a financial institution", True),
        ("The bat flew out of the cave at dusk", "He swung the bat and hit a home run", False),
        ("The bat flew out of the cave at dusk", "A nocturnal mammal with wings", True),
        ("Turn on the light please", "This suitcase is very light", False),
        ("Turn on the light please", "The room was bright and sunny", True),
        ("Children play in the park every afternoon", "The musician will play the violin tonight", False),
        ("Shakespeare wrote many famous plays", "Hamlet is a tragic drama performed on stage", True),
    ]
    all_texts = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))
    cache.get(all_texts)

    passed = 0
    for sent_a, sent_b, should_close in pairs:
        embs = cache.get_jepa_embeddings([sent_a, sent_b], encoder)
        sim = cosine_sim(embs[0], embs[1])
        is_correct = (sim > Config.fine_grained_close_threshold) if should_close else (sim < Config.fine_grained_far_threshold)
        if is_correct: passed += 1
        expected = "близки" if should_close else "далеки"
        status = "✓" if is_correct else "✗"
        print(f"  {status} sim={sim:.3f} (ожидается: {expected})")
    print_result("Тонкие смыслы", passed, len(pairs))
    return passed, len(pairs)


def test_unsupervised_clustering(cache, encoder, config):
    print("\n" + "=" * 70)
    print("🧪 ТЕСТ 6: КЛАСТЕРИЗАЦИЯ БЕЗ УЧИТЕЛЯ (Unsupervised Clustering)")
    print("=" * 70)
    texts = [
        "NASA launched a new satellite into orbit", "Mars rovers are exploring the red planet", "Black holes have immense gravitational pull", "The Milky Way contains billions of stars", "Astronauts trained for the space mission", "Telescopes observe distant galaxies",
        "The chef prepared a delicious pasta dish", "Fresh herbs enhance the flavor of any meal", "Baking requires precise measurements", "Olive oil is a staple in Mediterranean cooking", "The restaurant serves authentic sushi", "Seasoning meat properly is essential",
        "The soccer team won the championship", "Tennis requires quick reflexes and agility", "Marathon runners train for months", "The basketball game went into overtime", "Swimming is excellent cardiovascular exercise", "The Olympic Games bring nations together",
    ]
    true_labels = ([0] * 6) + ([1] * 6) + ([2] * 6)
    cache.get(texts)
    embs = cache.get_jepa_embeddings(texts, encoder)

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
    for perm in permutations([0, 1, 2]):
        mapped = np.array([perm[p] for p in predicted])
        acc = (mapped == np.array(true_labels)).mean()
        if acc > best_acc: best_acc = acc

    print(f"\n  🎯 Темы: Космос / Кулинария / Спорт")
    print(f"  📊 Accuracy: {best_acc*100:.1f}%")
    passed = 1 if best_acc >= Config.clustering_accuracy_threshold else 0
    print_result("Кластеризация", passed, 1, f"Точность: {best_acc*100:.1f}% (порог: {Config.clustering_accuracy_threshold})")
    return passed, 1


def test_hierarchical_similarity(cache, encoder, config):
    print("\n" + "=" * 70)
    print("🧪 ТЕСТ 7: ИЕРАРХИЯ ПОНЯТИЙ (Hierarchical Similarity)")
    print("=" * 70)
    hierarchies = [
        {"name": "Животные", "abstract": "Living organisms in the animal kingdom", "middle": "Domestic dogs are loyal companions", "specific": "The golden retriever puppy played fetch in the park"},
        {"name": "Технологии", "abstract": "Modern technology shapes our daily lives", "middle": "Smartphones have become essential communication devices", "specific": "The iPhone 15 features a titanium frame and USB-C port"},
        {"name": "Еда", "abstract": "Food provides essential nutrients for the body", "middle": "Italian cuisine is famous for pasta and pizza", "specific": "Spaghetti carbonara uses eggs, pecorino, guanciale, and black pepper"},
    ]
    all_texts = [t for h in hierarchies for t in [h["abstract"], h["middle"], h["specific"]]]
    cache.get(list(set(all_texts)))

    passed = 0
    for h in hierarchies:
        embs = cache.get_jepa_embeddings([h["abstract"], h["middle"], h["specific"]], encoder)
        sim_am = cosine_sim(embs[0], embs[1])
        sim_ms = cosine_sim(embs[1], embs[2])
        sim_as = cosine_sim(embs[0], embs[2])
        print(f"\n  🎯 Иерархия: {h['name']}")
        print(f"     sim(abstract, middle)   = {sim_am:.3f}")
        print(f"     sim(middle, specific)   = {sim_ms:.3f}")
        print(f"     sim(abstract, specific) = {sim_as:.3f}")
        is_correct = (sim_am > sim_as) and (sim_ms > sim_as)
        if is_correct: passed += 1
    print_result("Иерархия понятий", passed, len(hierarchies))
    return passed, len(hierarchies)


# ==========================================
# ГЛАВНАЯ ФУНКЦИЯ
# ==========================================
def main():
    print("🚀 ТЕСТИРОВАНИЕ ОБУЧЕННОЙ TEXT-JEPA МОДЕЛИ")
    print("=" * 70)
    cache, encoder, config = load_models()

    results = []
    results.append(("1. Парафразы", test_paraphrases(cache, encoder, config)))
    results.append(("2. Белая ворона", test_odd_one_out(cache, encoder, config)))
    results.append(("3. Устойчивость к шуму", test_noise_robustness(cache, encoder, config)))
    results.append(("4. Асимметричный поиск", test_asymmetric_retrieval(cache, encoder, config)))
    results.append(("5. Тонкие смыслы", test_fine_grained_senses(cache, encoder, config)))
    results.append(("6. Кластеризация", test_unsupervised_clustering(cache, encoder, config)))
    results.append(("7. Иерархия понятий", test_hierarchical_similarity(cache, encoder, config)))

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
    if total_pct >= 85: print("🏆 ОТЛИЧНО!")
    elif total_pct >= 70: print("✅ ХОРОШО!")
    elif total_pct >= 50: print("⚠️  УДОВЛЕТВОРИТЕЛЬНО")
    else: print("❌ ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНОЕ ОБУЧЕНИЕ")
    print("=" * 70)

    # Очистка
    cache.clear_cache()


if __name__ == "__main__":
    main()

