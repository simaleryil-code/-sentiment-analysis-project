import re
import requests


class OllamaCommentAnalyzer:
    def __init__(self, model="qwen2.5:3b"):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"

    def clean_comment(self, comment):
        if not comment:
            return ""

        text = str(comment).strip()

        if text.lower() in ["[sticker]", "sticker"]:
            return ""

        text_chars = re.findall(r"[a-zA-ZığüşöçİĞÜŞÖÇа-яА-Я0-9]", text)

        if len(text_chars) < 3:
            return ""

        return text

    def prepare_comments(self, comments, max_comment_count=200):
        cleaned_comments = []

        for comment in comments:
            cleaned = self.clean_comment(comment)

            if cleaned:
                cleaned_comments.append(cleaned)

            if len(cleaned_comments) >= max_comment_count:
                break

        return cleaned_comments

    def build_prompt(self, comments):
        comment_text = "\n".join(
            [f"{index + 1}. {comment}" for index, comment in enumerate(comments)]
        )

        return f"""
Sen profesyonel bir sosyal medya yorum analizi uzmanısın.

Aşağıdaki TikTok yorumlarını analiz edeceksin. Görevin yorumları düzeltmek, yeniden yazmak, önerilen cümle üretmek veya metni editlemek DEĞİLDİR. Görevin sadece yorum kitlesinin videoya verdiği tepkiyi analiz etmektir.

Kesin kurallar:
- Cevabın tamamen Türkçe olacak.
- İngilizce cevap verme.
- Yorumları tek tek düzeltme.
- Alternatif cümle önerme.
- "Bu metin şöyle yazılabilir", "daha açıklayıcı yazabilirsiniz", "şu cümleyi şöyle yapın" gibi ifadeler kullanma.
- Yorumları kopyalayıp listeleme.
- Sadece yorumların genel anlamını, duygu dağılımını ve izleyici algısını analiz et.
- Sticker, emoji, kısa ve anlamsız yorumları ayrı değerlendir.
- Emin olmadığın yerde kesin konuşma; "veriye göre", "yorum örneklerine göre", "sınırlı veriyle" gibi temkinli ifadeler kullan.
- Rapor ciddi, sade ve sunumda kullanılabilecek dilde olsun.

Rapor formatı tam olarak şu şekilde olsun:

# Rapor

## 1. Genel Kitle Algısı
Yorumlara göre izleyicilerin videoya genel yaklaşımını açıkla. Olumlu, olumsuz, nötr ve anlamsız yorumların genel tabloyu nasıl etkilediğini belirt.

## 2. Olumlu Yorumların Analizi
İnsanların videoda neyi beğenmiş olabileceğini açıkla. Övgü, destek, ilgi, eğlence veya beğeni ifadelerini değerlendir.

## 3. Olumsuz Yorumların Analizi
Eleştiri, alay, tepki, güvensizlik veya rahatsızlık içeren yorumlar varsa açıkla. Yoksa belirgin olumsuz tepkinin sınırlı olduğunu belirt.

## 4. Nötr ve Anlamsız Etkileşimler
Sticker, emoji, kısa tepki, etiketleme ve bağlamsız yorumların analiz değerini açıkla.

## 5. Öne Çıkan Temalar
Yorumlarda tekrar eden ana konuları madde madde yaz.

## 6. Duygu Dağılımı Yorumu
Yorum kitlesinin genel ruh halini değerlendir. Bu dağılımın içerik performansı açısından ne anlama geldiğini açıkla.

## 7. İçerik Üreticisi İçin Öneriler
Yorumlardan çıkarılabilecek uygulanabilir öneriler ver. Bu öneriler içerik stratejisiyle ilgili olsun; yorumları yeniden yazma önerisi verme.

## 8. Genel Sonuç
Kısa, net ve profesyonel bir genel sonuç yaz.

Analiz edilecek TikTok yorumları:
{comment_text}
"""

    def analyze_comments(self, comments, max_comment_count=200):
        selected_comments = self.prepare_comments(
            comments=comments,
            max_comment_count=max_comment_count
        )

        if not selected_comments:
            return (
                "# Rapor\n\n"
                "Yorumların büyük bölümü sticker, emoji veya anlamlı metin içermeyen kısa ifadelerden oluşuyor.\n\n"
                "Bu nedenle detaylı metinsel duygu analizi için yeterli veri bulunmamaktadır. "
                "Genel olarak bu yorumlar nötr veya düşük anlamlı etkileşim olarak değerlendirilebilir."
            )

        prompt = self.build_prompt(selected_comments)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.7,
                "num_predict": 1800
            }
        }

        response = requests.post(
            self.api_url,
            json=payload,
            timeout=240
        )

        response.raise_for_status()

        data = response.json()
        result = data.get("response", "").strip()

        if not result:
            return "LLM analiz sonucu alınamadı."

        return result
