import os
import re
import time
import requests
from datetime import datetime, timezone, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.request import urlretrieve
from dotenv import load_dotenv

load_dotenv()

# ======================== 설정 ==========================
PLAYBOARD_EMAIL = os.getenv("PLAYBOARD_EMAIL")
PLAYBOARD_PASSWORD = os.getenv("PLAYBOARD_PASSWORD")
YEAR = 2025
MONTH = 5

HD_MODE = True
USE_PERIOD_MODE = True  # True: period 기반, False: 날짜 버튼 클릭 방식


# ======================== 유틸 함수 ==========================
def create_folder(folder_name="playboard_thumbnails"):
    current_path = os.path.abspath(__file__)
    upper_path = os.path.dirname(os.path.dirname(current_path))  # 상위 폴더
    folder_path = os.path.join(upper_path, "data", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/115.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    return webdriver.Chrome(options=options)


def login_playboard(driver, email, password):
    login_url = "https://playboard.co/account/signin?retUri=%2F"
    driver.get(login_url)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "email"))
        ).send_keys(email)
        driver.find_element(By.NAME, "password").send_keys(password)
        login_button = driver.find_element(By.CSS_SELECTOR, "button.tbtn.tbtn--black")
        driver.execute_script("arguments[0].click();", login_button)
        WebDriverWait(driver, 10).until(EC.url_changes(login_url))
        print("✅ 로그인 완료")
    except Exception as e:
        print("❌ 로그인 실패:", e)


def wait_until_thumbnails_loaded(driver, timeout=15):
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.thumb.lazy-image"))
        )
        print("✅ 썸네일 요소 로드됨")
    except:
        print("⏱️ 썸네일 로딩 타임아웃")


def safe_page_load(driver, url, retries=3):
    for attempt in range(retries):
        try:
            driver.get(url)
            wait_until_thumbnails_loaded(driver, timeout=15)
            return True
        except:
            print(f"🔁 {attempt+1}번째 재시도 중...")
            time.sleep(3)
    print("❌ 페이지 로딩 실패")
    return False


def get_period_list_utc(year, month):
    UTC = timezone.utc
    result = []
    for day in range(1, 32):
        try:
            date_obj = datetime(year, month, day, tzinfo=UTC)
            label = date_obj.strftime("%Y.%m.%d")
            timestamp = int(date_obj.timestamp())
            result.append((label, timestamp))
        except:
            continue
    return result


def scroll_to_bottom_until_fully_loaded(
    driver, selector="div.thumb.lazy-image", max_wait=30
):
    last_count = 0
    same_count_retry = 0
    for _ in range(max_wait):
        driver.execute_script(
            "window.scrollTo(0, document.documentElement.scrollHeight);"
        )
        try:
            WebDriverWait(driver, 5).until(
                lambda d: len(d.find_elements(By.CSS_SELECTOR, selector)) > last_count
            )
        except:
            same_count_retry += 1
            if same_count_retry >= 3:
                break
        else:
            same_count_retry = 0
            last_count = len(driver.find_elements(By.CSS_SELECTOR, selector))
        print(f"🔽 현재 로드된 썸네일 div 수: {last_count}")
    print("✅ 스크롤 완료")


def extract_playboard_thumbnails(driver):
    thumbnails, seen_urls = [], set()
    elements = driver.find_elements(By.CSS_SELECTOR, "div.thumb.lazy-image")
    for el in elements:
        style = el.get_attribute("style")
        match = re.search(r"url\(['\"]?(.*?)['\"]?\)", style or "")
        url = match.group(1) if match else el.get_attribute("data-background-image")
        if url and url.startswith("//"):
            url = "https:" + url
        if url and url not in seen_urls:
            seen_urls.add(url)
            thumbnails.append(url)
    return thumbnails


def try_higher_quality(url, HD=False):
    if HD:
        quality = "maxresdefault"
        high_url = re.sub(
            r"(default|mqdefault|hqdefault|sddefault|maxresdefault)", quality, url
        )
        try:
            if requests.get(high_url, timeout=5).status_code == 200:
                return high_url, quality
            else:
                return None, None
        except:
            return None, None
    else:
        qualities = ["maxresdefault", "sddefault", "hqdefault", "mqdefault"]
        for quality in qualities:
            high_url = re.sub(
                r"(default|mqdefault|hqdefault|sddefault|maxresdefault)", quality, url
            )
            try:
                if requests.get(high_url, timeout=5).status_code == 200:
                    return high_url, quality
            except:
                continue
        return url, "unknown"


def download_images(image_urls, save_path, HD=False, label=None):
    base_folder = os.path.dirname(os.path.dirname(save_path))
    saved_ids_file = os.path.join(base_folder, "saved_urls.txt")

    saved_ids = set()
    existing_lines = []
    if os.path.exists(saved_ids_file):
        with open(saved_ids_file, "r") as f:
            for line in f:
                existing_lines.append(line.strip())
                url = line.strip().split()[-1]
                match = re.search(r"/vi/([a-zA-Z0-9_-]+)/", url)
                if match:
                    saved_ids.add(match.group(1))

    downloaded = 0
    new_lines = []
    quality_counter = {
        q: 0
        for q in ["maxresdefault", "sddefault", "hqdefault", "mqdefault", "unknown"]
    }

    for url in image_urls:
        high_url, quality = try_higher_quality(url, HD=HD)
        if not high_url:
            print(f"⏭️ 고화질 없음 → 스킵: {url}")
            continue

        match = re.search(r"/vi/([a-zA-Z0-9_-]+)/", high_url)
        video_id = match.group(1) if match else None
        if video_id in saved_ids:
            print(f"⚠️ 중복 ID: {video_id} → 스킵")
            continue

        filename = (
            f"{label}_{downloaded+1:03d}_{video_id}.jpg"
            if video_id
            else f"{label}_{downloaded+1:03d}.jpg"
        )
        filepath = os.path.join(save_path, filename)
        try:
            urlretrieve(high_url, filepath)
            print(f"✔ 저장됨: {filepath}")
            downloaded += 1
            saved_ids.add(video_id)
            new_lines.append(f"{label or 'unknown'} {high_url}")
            if not HD:
                quality_counter[quality] += 1
        except Exception as e:
            print(f"❌ 실패: {high_url} - {e}")

    with open(saved_ids_file, "w") as f:
        for line in existing_lines + new_lines:
            f.write(line + "\n")

    print(f"\n📦 저장 완료: {downloaded}장")
    if not HD:
        print("📊 화질별 저장 통계:")
        for q in quality_counter:
            print(f"  {q}: {quality_counter[q]}장")
    return quality_counter


def click_date_button_and_crawl(
    driver, year, month, base_path, hd_mode, total_quality_counter
):
    date_url = (
        "https://playboard.co/chart/video/most-viewed-all-videos-in-south-korea-daily"
    )
    driver.get(date_url)
    time.sleep(3)

    days = driver.find_elements(By.CSS_SELECTOR, "li.date-picker__item")
    for day_el in days:
        try:
            label = day_el.text.strip()
            if not label or label == "오늘":
                continue
            print(f"\n=== 📅 날짜 클릭: {label} ===")
            driver.execute_script("arguments[0].click();", day_el)
            time.sleep(2)
            scroll_to_bottom_until_fully_loaded(driver)
            thumbnails = extract_playboard_thumbnails(driver)
            date_folder = os.path.join(base_path, label)
            os.makedirs(date_folder, exist_ok=True)
            print(f"🎯 {label} 썸네일 수: {len(thumbnails)}")
            download_images(thumbnails, date_folder, HD=hd_mode, label=label)
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            continue


def main(hd_mode=True, use_period_mode=True):
    base_path = create_folder(
        "playboard_thumbnails_hd" if hd_mode else "playboard_thumbnails"
    )
    driver = setup_driver()
    login_playboard(driver, PLAYBOARD_EMAIL, PLAYBOARD_PASSWORD)

    total_quality_counter = {
        q: 0
        for q in ["maxresdefault", "sddefault", "hqdefault", "mqdefault", "unknown"]
    }

    if use_period_mode:
        period_list = get_period_list_utc(YEAR, MONTH)
        for label, timestamp in period_list:
            print(f"\n=== 🔍 날짜: {label} ===")
            date_folder = os.path.join(base_path, label)
            os.makedirs(date_folder, exist_ok=True)
            period_url = f"https://playboard.co/chart/video/most-viewed-all-videos-in-south-korea-daily?period={timestamp}"
            if not safe_page_load(driver, period_url):
                continue
            try:
                scroll_to_bottom_until_fully_loaded(driver)
                thumbnails = extract_playboard_thumbnails(driver)
                print(f"🎯 {label} 썸네일 수: {len(thumbnails)}")
                download_images(thumbnails, date_folder, HD=hd_mode, label=label)
            except Exception as e:
                print(f"❌ 처리 중 오류: {e}")
                continue
    else:
        click_date_button_and_crawl(
            driver, YEAR, MONTH, base_path, hd_mode, total_quality_counter
        )

    driver.quit()
    print("\n📈 전체 화질별 누적 통계:")
    for q in total_quality_counter:
        print(f"  {q}: {total_quality_counter[q]}장")


if __name__ == "__main__":
    main(HD_MODE, USE_PERIOD_MODE)
