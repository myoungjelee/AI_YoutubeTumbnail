# crawler.py

import time
import os
import re
from datetime import datetime
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def is_duplicate_image(video_id, save_dir):
    for filename in os.listdir(save_dir):
        if filename.endswith(".jpg") and video_id in filename:
            return True
    return False


def sanitize_filename(filename):
    """
    파일 이름으로 사용할 수 없는 문자 및 인코딩 문제를 일으킬 수 있는
    이모지, 특수 기호 등을 완벽하게 제거합니다.
    """
    # 1. 기본적으로 허용되지 않는 파일 시스템 문자 제거
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    # 2. 이모지를 포함한 대부분의 특수 기호를 제거 (한글, 영문, 숫자, 공백, 점, 밑줄, 하이픈만 허용)
    sanitized = re.sub(r"[^a-zA-Z0-9가-힣\s\._-]", "", sanitized).strip()
    # 3. 공백이 연속으로 오는 경우 하나로 합침
    sanitized = re.sub(r"\s+", " ", sanitized)
    return sanitized


def get_high_quality_thumbnail_url(video_id):
    """가능한 최고 화질의 썸네일 URL을 반환합니다."""
    urls_to_try = [
        f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
        f"https://i.ytimg.com/vi/{video_id}/hq720.jpg",
        f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
    ]
    for url in urls_to_try:
        try:
            response = requests.head(url, timeout=5)
            if response.status_code == 200:
                return url
        except requests.RequestException:
            continue
    return urls_to_try[-1]


def download_and_verify_image(url, path, title):
    """URL에서 이미지를 다운로드하고 성공 여부를 검증합니다."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        img_data = requests.get(url, headers=headers, timeout=10).content
        with open(path, "wb") as handler:
            handler.write(img_data)

        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"  [성공] 썸네일 저장 완료: {os.path.basename(path)}")
            return True
        else:
            print(f"  [실패] 썸네일 파일 생성 실패 또는 크기 0: {title}")
            return False
    except Exception as e:
        print(f"  [오류] 썸네일 다운로드 중 오류 발생: {e}")
        return False


def crawl_youtube_trending():
    """
    유튜브 인기 급상승 동영상의 썸네일과 정보를 크롤링합니다.
    성공 시 썸네일이 저장된 폴더 경로를 반환합니다.
    """
    youtube_trending_url = "https://www.youtube.com/feed/trending?bp=6gQJRkVleHBsb3Jl"

    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    base_folder = os.path.dirname(os.path.dirname(__file__))
    image_folder = os.path.join("data", "Youtube_Trending", f"{timestamp_str}")
    os.makedirs(image_folder, exist_ok=True)
    print(f"폴더 생성: {image_folder}")

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=ko_KR")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    driver = webdriver.Chrome(options=options)
    print("WebDriver 시작됨")

    try:
        driver.get(youtube_trending_url)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "ytd-video-renderer"))
        )

        last_height = driver.execute_script(
            "return document.documentElement.scrollHeight"
        )
        while True:
            driver.execute_script(
                "window.scrollTo(0, document.documentElement.scrollHeight);"
            )
            time.sleep(2)
            new_height = driver.execute_script(
                "return document.documentElement.scrollHeight"
            )
            if new_height == last_height:
                break
            last_height = new_height

        all_video_elements = driver.find_elements(By.CSS_SELECTOR, "ytd-video-renderer")
        print(f"\n총 {len(all_video_elements)}개의 동영상 발견. 데이터 추출 시작...")

        video_data = []
        for i, video_element in enumerate(all_video_elements):
            try:
                title_element = video_element.find_element(
                    By.CSS_SELECTOR, "a#video-title"
                )
                title = title_element.get_attribute("title")
                link = title_element.get_attribute("href")

                if not link or "watch?v=" not in link:
                    continue

                video_id = link.split("watch?v=")[1].split("&")[0]
                print(f"\n[{i + 1}/{len(all_video_elements)}] 처리 중: {title}")

                if is_duplicate_image(video_id, image_folder):
                    print(f"  ⚠️ 중복된 영상 ID → 다운로드 생략: {video_id}")
                    continue

                thumbnail_url = get_high_quality_thumbnail_url(video_id)
                print(f"  고화질 썸네일 URL 확보: {thumbnail_url}")

                today_str = datetime.now().strftime("%Y.%m.%d")
                image_filename = f"{today_str}_{i + 1:03d}_{video_id}.jpg"
                image_path = os.path.join(image_folder, image_filename)

                if download_and_verify_image(thumbnail_url, image_path, title):
                    video_data.append(
                        {
                            "rank": i + 1,
                            "title": title,
                            "link": link,
                            "thumbnail_file": image_filename,
                        }
                    )
            except Exception as e:
                print(f"  - 동영상 정보 처리 중 예상치 못한 오류: {e}")
                continue

        if video_data:
            df = pd.DataFrame(video_data)
            csv_path = os.path.join(
                base_folder, f"youtube_trending_rankings_{timestamp_str}.csv"
            )
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"\n데이터 저장 완료: {csv_path} ({len(video_data)}개 수집)")
            return image_folder
        else:
            print("\n수집된 유효한 데이터가 없습니다.")
            return None

    except Exception as e:
        print(f"크롤링 중 심각한 오류 발생: {e}")
        return None
    finally:
        driver.quit()
        print("WebDriver 종료됨")


if __name__ == "__main__":
    print("--- 크롤러 모듈 단독 테스트 실행 ---")
    result_folder = crawl_youtube_trending()
    if result_folder:
        print(f"\n[테스트 성공] 썸네일이 저장된 최종 경로: {result_folder}")
    else:
        print("\n[테스트 실패] 크롤링에 실패했거나 수집된 데이터가 없습니다.")
