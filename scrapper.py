import requests as REQUEST
import json as JSON
import argparse
from time import sleep
import datetime
import time
import csv
import re as RE
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class TikTokExtractor:
    def __init__(self, url, output="output.json", comment_count=None, file_type="json"):
        self.url = url
        self.output = output
        self.comment_count = comment_count
        self.file_type = file_type

        self.metadata = {
            "metadata": {},
            "comments": []
        }

        self.headers = {
            "Host": "www.tiktok.com",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
            "Referer": "https://www.tiktok.com/",
            "Connection": "keep-alive"
        }

    @staticmethod
    def convert(n):
        return str(datetime.timedelta(seconds=n))

    def extract_video_id(self):
        match = RE.search(r"/video/(\d+)", self.url)

        if match:
            return match.group(1)

        match = RE.search(r"(\d+)$", self.url)

        if match:
            return match.group(1)

        raise ValueError("Video ID bulunamadı. Geçerli bir TikTok video bağlantısı girin.")

    def extract_metadata(self):
        """
        TikTok video üst bilgilerini gerçek tarayıcı render'ı üzerinden alır.
        F12'de görünen data-e2e alanlarını Playwright ile okur.
        """

        import re
        import time
        from playwright.sync_api import sync_playwright

        def extract_video_id_from_url(url):
            match = re.search(r"/video/(\d+)", url)
            if match:
                return match.group(1)
            return "Bilinmiyor"

        def extract_username_from_url(url):
            match = re.search(r"tiktok\.com/@([^/]+)/video/", url)
            if match:
                return match.group(1)
            return "Bilinmiyor"

        def clean_count(value):
            if value is None:
                return "0"

            value = str(value).strip()

            if value == "":
                return "0"

            return value

        def read_first_text(page, selectors, default="BULUNAMADI"):
            for selector in selectors:
                try:
                    locator = page.locator(selector).first

                    if locator.count() > 0:
                        text = locator.inner_text(timeout=5000).strip()

                        if text:
                            return text
                except Exception:
                    pass

            return default

        video_id = extract_video_id_from_url(self.url)
        url_username = extract_username_from_url(self.url)

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=False,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage"
                ]
            )

            context = browser.new_context(
                viewport={"width": 1366, "height": 768},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/121.0.0.0 Safari/537.36"
                ),
                locale="tr-TR"
            )

            page = context.new_page()

            try:
                page.goto(self.url, wait_until="domcontentloaded", timeout=60000)

                # TikTok değerleri JavaScript ile sonradan bastığı için bekliyoruz.
                page.wait_for_timeout(10000)

                like_count = read_first_text(
                    page,
                    [
                        '[data-e2e="like-count"]',
                        'strong[data-e2e="like-count"]'
                    ],
                    default="0"
                )

                comment_count = read_first_text(
                    page,
                    [
                        '[data-e2e="comment-count"]',
                        'strong[data-e2e="comment-count"]'
                    ],
                    default="0"
                )

                share_count = read_first_text(
                    page,
                    [
                        '[data-e2e="share-count"]',
                        'strong[data-e2e="share-count"]'
                    ],
                    default="0"
                )

                description = read_first_text(
                    page,
                    [
                        '[data-e2e="browse-video-desc"]',
                        '[data-e2e="video-desc"]',
                        'h1[data-e2e="browse-video-desc"]'
                    ],
                    default="Video açıklaması alınamadı."
                )

                username = read_first_text(
                    page,
                    [
                        '[data-e2e="browse-username"]',
                        '[data-e2e="video-author-uniqueid"]',
                        'a[data-e2e="browse-username"]'
                    ],
                    default=url_username
                )

                if username == "BULUNAMADI":
                    username = url_username

                nickname = read_first_text(
                    page,
                    [
                        '[data-e2e="browser-nickname"]',
                        '[data-e2e="video-author-nickname"]'
                    ],
                    default=username
                )

                if nickname == "BULUNAMADI":
                    nickname = username

                self.metadata["metadata"] = {
                    "idVideo": video_id,
                    "uniqueId": username,
                    "nickname": nickname,
                    "description": description,
                    "totalLike": clean_count(like_count),
                    "totalComment": clean_count(comment_count),
                    "totalShare": clean_count(share_count),
                    "createTime": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": "Bilinmiyor",
                    "metadataStatus": "Video bilgisi Playwright ile sayfa üzerinden alındı."
                }

            finally:
                context.close()
                browser.close()


    def fetch_comment_page(self, video_id, cursor):
        api_url = (
            "https://www.tiktok.com/api/comment/list/"
            f"?aid=1988"
            f"&app_language=en"
            f"&app_name=tiktok_web"
            f"&aweme_id={video_id}"
            f"&count=50"
            f"&cursor={cursor}"
            f"&os=windows"
            f"&region=TR"
            f"&screen_height=768"
            f"&screen_width=1366"
            f"&user_is_login=false"
        )

        last_error = None

        for attempt in range(1, 4):
            try:
                response = REQUEST.get(
                    api_url,
                    headers=self.headers,
                    verify=False,
                    timeout=20
                )

                return JSON.loads(response.text)

            except Exception as error:
                last_error = error
                sleep(attempt * 2)

        raise ConnectionError(f"TikTok bağlantısı kesildi. Cursor: {cursor}. Hata: {last_error}")

    def extract_comments(self):
        video_id = self.metadata["metadata"].get("idVideo")

        if not video_id:
            raise ValueError("Yorum çekmek için video ID bulunamadı.")

        cursor = 0
        empty_page_count = 0

        while True:
            try:
                data = self.fetch_comment_page(video_id, cursor)
            except Exception as error:
                self.metadata["metadata"]["fetchStatus"] = (
                    f"TikTok bağlantıyı kesti. O ana kadar çekilen yorumlar kaydedildi. Detay: {error}"
                )
                break

            if data.get("status_code") not in [0, None]:
                self.metadata["metadata"]["fetchStatus"] = (
                    f"TikTok API hata döndürdü. O ana kadar çekilen yorumlar kaydedildi. Detay: {data}"
                )
                break

            comments = data.get("comments", [])

            if len(comments) == 0:
                empty_page_count += 1

                if empty_page_count >= 2:
                    self.metadata["metadata"]["fetchStatus"] = "TikTok daha fazla yorum döndürmedi."
                    break

                cursor += 50
                sleep(1)
                continue

            empty_page_count = 0

            for comment in comments:
                if self.comment_count and len(self.metadata["comments"]) >= self.comment_count:
                    break

                user = comment.get("user", {})
                username = user.get("nickname", "Anonim")
                text = comment.get("text", "")

                if text:
                    self.metadata["comments"].append({
                        "username": username,
                        "comment": text
                    })

            self.metadata["metadata"]["lastCursor"] = cursor
            self.metadata["metadata"]["totalComment"] = len(self.metadata["comments"])

            if self.comment_count and len(self.metadata["comments"]) >= self.comment_count:
                self.metadata["metadata"]["fetchStatus"] = "Belirlenen maksimum yorum sayısına ulaşıldı."
                break

            if data.get("has_more") == 0:
                self.metadata["metadata"]["fetchStatus"] = "Tüm erişilebilir yorumlar çekildi."
                break

            cursor += 50
            sleep(1)

        self.metadata["metadata"]["totalComment"] = len(self.metadata["comments"])

        if len(self.metadata["comments"]) == 0:
            raise ValueError(
                "Yorum çekilemedi. Video yoruma kapalı olabilir, TikTok isteği sınırlamış olabilir "
                "veya bu video için yorum API boş dönüyor olabilir."
            )

    def save_to_file(self):
        if self.file_type == "json":
            with open(self.output, "w", encoding="utf-8") as file:
                JSON.dump(self.metadata, file, ensure_ascii=False, indent=4)

        elif self.file_type == "csv":
            with open(self.output, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["username", "comment"])

                for comment in self.metadata["comments"]:
                    writer.writerow([comment["username"], comment["comment"]])

        elif self.file_type == "txt":
            with open(self.output, "w", encoding="utf-8") as file:
                for comment in self.metadata["comments"]:
                    file.write(f"{comment['username']}: {comment['comment']}\n")

        else:
            raise ValueError("Desteklenmeyen dosya tipi.")

    def run(self):
        self.extract_metadata()
        self.extract_comments()
        self.save_to_file()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TikTok video yorumlarını çıkarır.")
    parser.add_argument("-u", "--url", type=str, required=True, help="TikTok video URL")
    parser.add_argument("-o", "--output", type=str, default="output.json", help="Çıktı dosyası")
    parser.add_argument("-c", "--comment", type=int, help="Çekilecek maksimum yorum sayısı")
    parser.add_argument("-f", "--file-type", type=str, default="json", help="json, csv veya txt")

    args = parser.parse_args()

    try:
        extractor = TikTokExtractor(
            url=args.url,
            output=args.output,
            comment_count=args.comment,
            file_type=args.file_type
        )
        extractor.run()
        print(f"Veriler başarıyla kaydedildi: {args.output}")

    except Exception as e:
        print(f"Hata: {e}")