# TikTok Yorum Analiz Sistemi

Bu proje, bir TikTok videosuna gelen yorumları çekip bu yorumları yapay zekâ destekli olarak analiz eden bir web uygulamasıdır.

Kullanıcı uygulamaya bir TikTok video bağlantısı girer. Sistem videodaki yorumları toplar, yorumları olumlu, olumsuz, nötr, anlamsız, spam veya toksik gibi sınıflara ayırır. Sonuçlar tablo, grafik ve LLM tarafından oluşturulan genel analiz raporu şeklinde gösterilir.

---

## 1. Projenin Amacı

Bu projenin amacı TikTok videolarına gelen yorumların genel duygu durumunu analiz etmektir.

Sistem şu sorulara cevap arar:

- İnsanlar videoyu beğenmiş mi?
- Yorumlarda olumsuz tepki var mı?
- Yorumların çoğu anlamlı mı, yoksa emoji ve sticker gibi düşük anlamlı etkileşimlerden mi oluşuyor?
- İçerik üreticisi bu yorumlardan nasıl bir sonuç çıkarabilir?
- Genel izleyici algısı olumlu mu, olumsuz mu, nötr mü?

Kısa anlatım:

> Bu uygulama TikTok yorumlarını otomatik okuyup insanların videoya nasıl tepki verdiğini anlamaya yarar.

---

## 2. Proje Ne İşe Yarar?

Bir TikTok videosunda yüzlerce veya binlerce yorum olabilir. Bu yorumları tek tek okumak hem zor hem de zaman alıcıdır.

Bu uygulama yorumları otomatik olarak analiz eder ve kullanıcıya anlaşılır bir özet sunar.

Uygulama şunları gösterir:

- Analiz edilen yorum sayısı
- Olumlu yorum sayısı
- Olumsuz yorum sayısı
- Nötr yorum sayısı
- Her yorumun duygu sınıfı
- Her yorumun kategorisi
- Yapay zekânın güven skoru
- Yorumun neden o şekilde sınıflandırıldığı
- Genel LLM analiz raporu

---

## 3. Kullanılan Teknolojiler

### Python

Projenin ana programlama dilidir. Yorum çekme, analiz etme, dosya okuma ve yapay zekâ bağlantıları Python ile yapılır.

### Streamlit

Python ile web arayüzü oluşturmayı sağlar. Bu projede uygulamanın ekranı Streamlit ile hazırlanmıştır.

Kullanıcı Streamlit ekranından:

- TikTok linki girer
- Analizi başlatır
- Sonuçları tablo, grafik ve rapor olarak görür

### NLTK

Doğal dil işleme işlemleri için kullanılır. Yorumların kelimelere ayrılması ve bazı metin temizleme işlemlerinde yardımcı olur.

### TextBlob ve VADER

Projenin ilk halinde klasik duygu analizi için kullanılmıştır. Fakat Türkçe ve sosyal medya yorumlarında yetersiz kalabildikleri için proje LLM destekli hale getirilmiştir.

### Ollama

Yerel bilgisayarda büyük dil modeli çalıştırmayı sağlar. Bu projede ücretli API kullanmadan yapay zekâ analizi yapmak için kullanılmıştır.

### Qwen2.5:3B

Ollama üzerinde çalışan yerel LLM modelidir. Yorumları sınıflandırmak ve genel rapor oluşturmak için kullanılır.

LLM, Large Language Model anlamına gelir. Türkçesi büyük dil modelidir.

---

## 4. Projenin Genel Çalışma Mantığı

Proje şu sırayla çalışır:

```text
TikTok video linki girilir
        ↓
Yorumlar çekilir
        ↓
Yorumlar temizlenir
        ↓
Yorumlar sınıflandırılır
        ↓
Sonuçlar tabloya aktarılır
        ↓
Grafik oluşturulur
        ↓
LLM genel rapor üretir
```

---

## 5. Kullanıcı Uygulamayı Nasıl Kullanır?

1. Uygulama başlatılır.
2. Kullanıcı TikTok video bağlantısını giriş alanına yazar.
3. **Yorumları Analiz Et** butonuna basar.
4. Sistem yorumları çekmeye başlar.
5. Yorumlar analiz edilir.
6. Sonuçlar ekranda gösterilir.

Örnek TikTok linki:

```text
https://www.tiktok.com/@kullanici/video/123456789
```

---

## 6. TikTok Yorumları Nasıl Çekiliyor?

Yorum çekme işlemi `scrapper.py` dosyasındaki `TikTokExtractor` sınıfı ile yapılır.

Bu sınıf:

1. Girilen TikTok video bağlantısını alır.
2. Video ID bilgisini bulur.
3. TikTok yorumlarını çekmeye çalışır.
4. Sonuçları `output.json` dosyasına yazar.

`output.json` içinde iki ana veri bulunur:

### Video Bilgileri

- Video ID
- Kullanıcı adı
- Açıklama
- Beğeni sayısı
- Yorum sayısı
- Paylaşım sayısı
- Oluşturulma tarihi

### Yorum Bilgileri

- Yorumu yazan kullanıcı
- Yorum metni

---

## 7. Neden Yorum Sayısı Sınırlı?

TikTok videolarında bazen çok fazla yorum olabilir. Tüm yorumları sınırsız çekmeye çalışmak bazı sorunlar çıkarabilir:

- TikTok bağlantıyı kesebilir
- Uygulama yavaşlayabilir
- Yerel LLM çok uzun sürede cevap verebilir
- Bilgisayar zorlanabilir
- Uygulama donabilir

Bu yüzden projede yorum çekme ve analiz sınırı kontrollü tutulur.

Sunumda şöyle anlatılabilir:

> Sen burayı anlatırkenb sınırlı olduğunu sormadıkça söyleme LLM analizi uzun süreceği için sunumda az yorumlarla denedim dersin ama  3000 yoruma kadar denedim çalışıyor uzun sürüyor işte. 

---

## 8. Yorumlar Nasıl Analiz Ediliyor?

Yorum analizi iki aşamalıdır.

### 8.1. Ön Sınıflandırma

Bazı yorumları LLM’e göndermeye gerek yoktur. Örneğin:

```text
[Sticker]
😂😂😂
❤️❤️❤️
@
a
```

Bu yorumlar anlamlı metin içermez. Sistem bunları doğrudan **Anlamsız** veya **Düşük Anlam** olarak sınıflandırır.

Bu sayede model gereksiz yorumlarla uğraşmaz ve sistem hızlanır.

### 8.2. LLM ile Sınıflandırma

Anlamlı yorumlar Ollama üzerinde çalışan Qwen2.5:3B modeline gönderilir.

Model her yorum için şu bilgileri üretir:

- Duygu
- Kategori
- Güven skoru
- Gerekçe

Örnek:

```text
Yorum: çok güzel olmuş
Duygu: Olumlu
Kategori: Beğeni
Güven: 0.88
Gerekçe: Yorum beğeni veya olumlu duygu ifade eden kelimeler içeriyor.
```

Başka bir örnek:

```text
Yorum: berbat olmuş
Duygu: Olumsuz
Kategori: Eleştiri
Güven: 0.86
Gerekçe: Yorum memnuniyetsizlik veya olumsuz algı ifade ediyor.
```

---

## 9. Duygu Sınıfları

Projede yorumlar şu sınıflara ayrılır:

### Olumlu

Beğeni, destek veya övgü içeren yorumlardır.

Örnek:

```text
çok güzel olmuş
harika enerji
efsane
```

### Olumsuz

Eleştiri, memnuniyetsizlik veya negatif tepki içeren yorumlardır.

Örnek:

```text
berbat olmuş
fake
ne alaka
```

### Nötr

Net olumlu veya olumsuz duygu taşımayan yorumlardır.

Örnek:

```text
nerede bu
kim bu
```

### Anlamsız

Metinsel anlamı düşük olan yorumlardır.

Örnek:

```text
[Sticker]
😂😂😂
❤️❤️
```

### Spam

Reklam, tekrar veya düşük kaliteli otomatik yorum gibi algılanabilecek yorumlardır.

### Toksik

Hakaret, küfür veya saldırgan dil içeren yorumlardır.

---

## 10. Tabloda Ne Gösteriliyor?

Analizden sonra ekranda detaylı yorum tablosu görünür.

Tabloda şu kolonlar bulunur:

### Kullanıcı

Yorumu yazan kişinin kullanıcı adı.

### Yorum

Kullanıcının yazdığı yorum.

### Duygu

Yorumun genel duygu sınıfı.

Örnek:

```text
Olumlu
Olumsuz
Nötr
Anlamsız
```

### Kategori

Duygunun daha açıklayıcı alt sınıfıdır.

Örnek:

```text
Beğeni
Eleştiri
Düşük Anlam
Spam
Diğer
```

### Güven

Modelin verdiği karardan ne kadar emin olduğunu gösterir. 0 ile 1 arasında değer alır.

Örnek:

```text
0.88
```

Bu değer modelin karardan büyük ölçüde emin olduğunu gösterir.

### Gerekçe

Yorumun neden o sınıfa alındığını kısa şekilde açıklar.

Örnek:

```text
Yorum beğeni veya olumlu duygu ifade eden kelimeler içeriyor.
```

---

## 11. Grafik Ne Gösteriyor?

Grafik yorumların genel duygu dağılımını gösterir.

Örneğin:

- Olumlu yorumlar fazlaysa izleyici tepkisi genelde iyidir.
- Olumsuz yorumlar fazlaysa içerik tepki çekmiş olabilir.
- Nötr veya anlamsız yorumlar fazlaysa etkileşim vardır ama yorumların anlamı zayıftır.


---

## 12. LLM Genel Analiz Raporu Nedir?

Tablo her yorumu ayrı ayrı gösterir. LLM genel analiz raporu ise yorumların tamamına daha geniş açıdan bakar.

Bu raporda sistem şu konuları açıklar:

1. Genel kitle algısı
2. Olumlu yorumların analizi
3. Olumsuz yorumların analizi
4. Nötr ve anlamsız yorumların analizi
5. Öne çıkan temalar
6. Duygu dağılımı yorumu
7. İçerik üreticisi için öneriler
8. Genel sonuç

Sunumda şöyle anlatılabilir:

> Tablo bölümünde yorumlar tek tek sınıflandırılır. LLM genel analiz raporu bölümünde ise yorumların tamamı değerlendirilir ve genel bir sonuç çıkarılır.

---

## 13. Projedeki Ana Dosyalar

### sentiment_analysis.py

Ana uygulama dosyasıdır.

Görevleri:

- Streamlit arayüzünü oluşturur
- Kullanıcıdan TikTok linki alır
- Yorum çekme işlemini başlatır
- Analiz sonuçlarını ekranda gösterir
- Tablo, grafik ve raporu yönetir

### scrapper.py

TikTok yorumlarını çeken dosyadır.

Görevleri:

- Video linkinden video ID alır
- TikTok yorumlarını çeker
- Verileri JSON formatında kaydeder

### llm_comment_classifier.py

Yorumları tek tek sınıflandıran dosyadır.

Görevleri:

- Sticker ve emoji yorumlarını ayıklar
- Bazı yorumları kurallarla sınıflandırır
- Anlamlı yorumları LLM’e gönderir
- Duygu, kategori, güven ve gerekçe üretir

### llm_analyzer.py

Genel LLM raporu üreten dosyadır.

Görevleri:

- Yorumları topluca değerlendirir
- Türkçe genel analiz raporu oluşturur

### requirements.txt

Projede kullanılan Python paketlerini içerir.

---

## 14. Şebnem Hoca Sorarsa Verilecek Cevaplar

### Bu proje ne yapıyor?

Bu proje, TikTok yorumlarını çekip yorumların olumlu, olumsuz, nötr, anlamsız, spam veya toksik olup olmadığını analiz ediyor. Sonuçları tablo, grafik ve rapor olarak gösteriyor.

### Yorumları nasıl alıyorsun?

Kullanıcı TikTok video linki giriyor. Sistem bu linkten video ID bilgisini alıyor ve yorumları çekiyor. Çekilen veriler JSON formatında işleniyor.

### Duygu analizi nasıl yapılıyor?

Önce çok kısa, sticker veya sadece emoji olan yorumlar ayrılıyor. Daha sonra anlamlı yorumlar yerel LLM modeliyle sınıflandırılıyor. Model her yorum için duygu, kategori, güven skoru ve gerekçe üretiyor.

### LLM nedir?

LLM büyük dil modeli demektir. İnsan dilini anlayıp yorumlayabilen yapay zekâ modelidir. Bu projede yorumları daha anlamlı analiz etmek için kullanılmıştır.

### Neden Ollama kullandın?

Çünkü Ollama yerel bilgisayarda ücretsiz model çalıştırmayı sağlar. Bu sayede ücretli API kullanmadan yapay zekâ analizi yapılabilir.

### Neden tüm yorumlar analiz edilmiyor?

Çünkü TikTok videolarında çok fazla yorum olabilir. Tüm yorumları sınırsız analiz etmek sistemi yavaşlatabilir veya bağlantı sorunlarına sebep olabilir. Bu yüzden sistem belirli bir limit ile çalışır.

### Sadece emoji olan yorumlar nasıl değerlendiriliyor?

Sadece emoji veya sticker olan yorumlar metinsel anlam taşımadığı için anlamsız veya düşük anlamlı etkileşim olarak sınıflandırılır.

### Proje gerçek hayatta nerede kullanılabilir?

İçerik üreticileri, sosyal medya yöneticileri ve markalar videolarına gelen yorumları hızlıca analiz etmek için kullanabilir.

### Sistem hata yapabilir mi?

Evet. Duygu analizi yüzde yüz kesin değildir. Çünkü bazı yorumlar ironi, argo, emoji veya bağlam gerektiren ifadeler içerebilir. Bu yüzden sistem sonuçları destekleyici analiz olarak sunar.

---

## 15. Projenin Güçlü Yönleri

- Kullanımı kolaydır.
- Web arayüzü vardır.
- Yorumları tablo ve grafikle gösterir.
- Yapay zekâ destekli analiz yapar.
- Yerel model kullandığı için ücretsiz çalışabilir.
- Her yorum için gerekçe üretir.
- Genel analiz raporu oluşturur.
- Gerçek bir sosyal medya kullanım senaryosuna sahiptir.

---

## 16. Projenin Sınırlamaları

- TikTok bazen yorum çekmeyi engelleyebilir.
- Çok fazla yorum çekmek zaman alabilir.
- Yerel LLM küçük model olduğu için bazen hatalı yorumlayabilir.
- Emoji ve ironi içeren yorumları anlamak zor olabilir.
- Analiz sonuçları kesin gerçek değil, yapay zekâ tahminidir.
- Sistem performans için belirli yorum limitleriyle çalışır.

---

## 17. Sunumda Kullanılabilecek Kısa Açıklama

Bu proje, TikTok videolarına gelen yorumları analiz etmek için geliştirilmiş bir web uygulamasıdır. Kullanıcı bir TikTok video bağlantısı girer. Sistem bu videodaki yorumları çeker, yorumları yapay zekâ destekli olarak olumlu, olumsuz, nötr, anlamsız, spam veya toksik gibi sınıflara ayırır. Sonuçlar tablo, grafik ve genel analiz raporu olarak gösterilir. Projede Python, Streamlit, Ollama ve yerel LLM modeli kullanılmıştır. Amaç, sosyal medya yorumlarını tek tek okumadan izleyici tepkisini hızlıca anlayabilmektir.

---

## 18. Teknik Akış Özeti

```text
1. Kullanıcı TikTok linkini girer.
2. Streamlit arayüzü bu linki alır.
3. TikTokExtractor yorumları çeker.
4. Yorumlar output.json dosyasına yazılır.
5. Python bu JSON dosyasını okur.
6. Yorumlar temizlenir.
7. llm_comment_classifier.py yorumları sınıflandırır.
8. sentiment_analysis.py sonuçları tabloya ve grafiğe dönüştürür.
9. llm_analyzer.py genel rapor oluşturur.
10. Kullanıcı sonuçları web arayüzünde görür.
```

---

## 19. Proje Tek Cümleyle

> TikTok Yorum Analiz Sistemi, bir TikTok videosundaki yorumları çekip yapay zekâ destekli olarak analiz eden ve sonuçları anlaşılır şekilde sunan bir sosyal medya duygu analizi uygulamasıdır.
