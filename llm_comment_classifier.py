import json
import re
import requests


class OllamaCommentClassifier:
    def __init__(self, model="qwen2.5:3b"):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"

        self.positive_words = [
            "güzel", "guzel", "harika", "mükemmel", "mukemmel", "efsane",
            "iyi", "başarılı", "basarili", "süper", "super", "tatlı", "tatli",
            "beğendim", "begendim", "seviyorum", "sevdim", "kaliteli",
            "enerji", "yakışmış", "yakismis", "çok iyi", "cok iyi",
            "çok güzel", "cok guzel", "eline sağlık", "eline saglik"
        ]

        self.negative_words = [
            "kötü", "kotu", "berbat", "rezalet", "saçma", "sacma",
            "fake", "yalan", "boş", "bos", "olmamış", "olmamis",
            "ne alaka", "alakasız", "alakasiz", "cringe", "iğrenç", "igrenc",
            "beğenmedim", "begenmedim", "vasat", "çöp", "cop"
        ]

        self.toxic_words = [
            "salak", "aptal", "mal", "gerizekalı", "gerizekali",
            "orospu", "piç", "pic", "siktir", "amk", "aq"
        ]

    def normalize(self, comment):
        return str(comment or "").strip().lower()

    def is_meaningless_comment(self, comment):
        if not comment:
            return True

        text = self.normalize(comment)

        if text in ["[sticker]", "sticker"]:
            return True

        text_chars = re.findall(r"[a-zA-ZığüşöçİĞÜŞÖÇ0-9]", text)

        if len(text_chars) < 3:
            return True

        return False

    def contains_any(self, text, words):
        return any(word in text for word in words)

    def classify_rule_based(self, comment):
        text = self.normalize(comment)

        if self.is_meaningless_comment(comment):
            return {
                "sentiment": "Anlamsız",
                "category": "Düşük Anlam",
                "confidence": 0.95,
                "reason": "Yorum sticker, emoji veya anlamlı metin içermeyen kısa bir ifade olduğu için anlamsız sınıfına alındı."
            }

        if self.contains_any(text, self.toxic_words):
            return {
                "sentiment": "Toksik",
                "category": "Hakaret veya Toksik Dil",
                "confidence": 0.9,
                "reason": "Yorum hakaret veya toksik dil içeren ifade barındırıyor."
            }

        if self.contains_any(text, self.positive_words):
            return {
                "sentiment": "Olumlu",
                "category": "Beğeni",
                "confidence": 0.88,
                "reason": "Yorum beğeni veya olumlu duygu ifade eden kelimeler içeriyor."
            }

        if self.contains_any(text, self.negative_words):
            return {
                "sentiment": "Olumsuz",
                "category": "Eleştiri",
                "confidence": 0.86,
                "reason": "Yorum eleştiri, memnuniyetsizlik veya olumsuz algı ifade eden kelimeler içeriyor."
            }

        return None

    def build_prompt(self, comments):
        comment_text = "\n".join(
            [f"{index + 1}. {comment}" for index, comment in enumerate(comments)]
        )

        return f"""
Sen Türkçe cevap veren bir sosyal medya yorum sınıflandırma sistemisin.

Aşağıdaki TikTok yorumlarını tek tek sınıflandır.

ÇOK ÖNEMLİ:
- Cevabın SADECE geçerli JSON array olacak.
- Markdown kullanma.
- JSON dışında açıklama yazma.
- Her yorum için mutlaka bir sonuç üret.
- index değeri sana verilen yorum numarasıyla aynı olmalı.
- Cevap dili Türkçe olacak.
- "harika", "güzel", "çok iyi", "efsane", "mükemmel" gibi ifadeler Olumlu sayılır.
- "berbat", "kötü", "fake", "ne alaka", "saçma", "rezalet" gibi ifadeler Olumsuz sayılır.
- Sadece emoji, sticker veya anlamsız kısa yorumlar Anlamsız sayılır.

Duygu seçenekleri:
- Olumlu
- Olumsuz
- Nötr
- Anlamsız
- Spam
- Toksik

Kategori seçenekleri:
- Beğeni
- Eleştiri
- Nötr Tepki
- Düşük Anlam
- Spam
- Hakaret veya Toksik Dil
- Diğer

JSON formatı:
[
  {{
    "index": 1,
    "sentiment": "Olumlu",
    "category": "Beğeni",
    "confidence": 0.85,
    "reason": "Yorum açık biçimde beğeni ifade ediyor."
  }}
]

Yorumlar:
{comment_text}
"""

    def extract_json(self, text):
        if not text:
            return []

        text = text.strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\[.*\]", text, re.S)

        if not match:
            return []

        try:
            return json.loads(match.group(0))
        except Exception:
            return []

    def classify_batch_with_llm(self, comments):
        prompt = self.build_prompt(comments)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.05,
                "top_p": 0.7,
                "num_predict": 2200
            }
        }

        response = requests.post(
            self.api_url,
            json=payload,
            timeout=240
        )

        response.raise_for_status()

        data = response.json()
        raw_response = data.get("response", "")

        parsed = self.extract_json(raw_response)

        if not isinstance(parsed, list):
            return []

        return parsed

    def classify_comments(self, comments, batch_size=10, max_comment_count=100):
        limited_comments = comments[:max_comment_count]

        final_results = []
        llm_queue = []
        llm_original_indexes = []

        for original_index, comment in enumerate(limited_comments, start=1):
            rule_result = self.classify_rule_based(comment)

            if rule_result:
                final_results.append({
                    "index": original_index,
                    "comment": comment,
                    **rule_result
                })
            else:
                llm_queue.append(comment)
                llm_original_indexes.append(original_index)

        for start in range(0, len(llm_queue), batch_size):
            batch_comments = llm_queue[start:start + batch_size]
            batch_indexes = llm_original_indexes[start:start + batch_size]

            llm_results = self.classify_batch_with_llm(batch_comments)

            llm_result_map = {}

            for item in llm_results:
                if not isinstance(item, dict):
                    continue

                try:
                    local_index = int(item.get("index", 0))
                except Exception:
                    continue

                llm_result_map[local_index] = item

            for local_index, original_index in enumerate(batch_indexes, start=1):
                comment = limited_comments[original_index - 1]
                item = llm_result_map.get(local_index)

                if not item:
                    final_results.append({
                        "index": original_index,
                        "comment": comment,
                        "sentiment": "Nötr",
                        "category": "Diğer",
                        "confidence": 0.3,
                        "reason": "LLM bu yorum için güvenilir sınıflandırma döndürmedi."
                    })
                    continue

                final_results.append({
                    "index": original_index,
                    "comment": comment,
                    "sentiment": item.get("sentiment", "Nötr"),
                    "category": item.get("category", "Diğer"),
                    "confidence": item.get("confidence", 0.5),
                    "reason": item.get("reason", "Kısa gerekçe üretilemedi.")
                })

        final_results.sort(key=lambda x: x["index"])

        return final_results
