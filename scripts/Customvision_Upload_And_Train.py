"""
📌 목적: COCO 라벨 데이터를 Azure Custom Vision에 업로드하고,
        새로운 Iteration을 학습시키며 퍼블리시까지 자동화하는 스크립트

🧩 주요 기능 및 워크플로우:
1. 📂 COCO 파일 로드 및 이미지 수 확인
2. 🏷️ Azure에 존재하는 태그 목록을 조회하여 매핑
3. 🔁 COCO 포맷을 Azure 업로드 포맷으로 변환
4. 📤 이미지 + 라벨을 Azure에 업로드 (base64 인코딩 방식)
5. 🧠 새로운 Iteration 이름 자동 생성 후 학습 트리거
6. ⏳ 학습 완료까지 상태 모니터링
7. 🚀 학습이 완료되면 Iteration 퍼블리시 수행

🔧 사전 준비사항:
- 프로젝트 ID(나머진 다 같은 리소스 그룹이라 동일)

"""

import os, json, base64, requests, time
import Customvision_Predict_To_Labelstudio as ptl
from dotenv import load_dotenv

load_dotenv()
# =========================
# 🔧 설정값
# =========================
USE_ADVANCED_TRAINING = True  # Quick, Advanced 학습 조절 플래그

AZURE_TRAINING_KEY = os.getenv("AZURE_TRAINING_KEY")
AZURE_TRAINING_ENDPOINT = os.getenv("AZURE_TRAINING_ENDPOINT")
AZURE_PREDICTION_KEY = os.getenv("AZURE_PREDICTION_KEY")
AZURE_PREDICTION_RESOURCE_ID = os.getenv("AZURE_PREDICTION_RESOURCE_ID")
AZURE_TRAINING_PROJECT_ID = os.getenv("AZURE_TRAINING_PROJECT_ID")

TRAIN_HEADERS = {
    "Training-Key": AZURE_TRAINING_KEY,
    "Content-Type": "application/json",
}

BASE_DIR = ptl.BASE_DIR
IMAGE_FOLDER = ptl.IMAGE_FOLDER
COCO_FILE_PATH = os.path.join(BASE_DIR, "result.json")


# =========================
# 📛 Iteration 이름 자동 생성
# =========================
def get_next_iteration_name():
    """
    현재 프로젝트의 Iteration 목록을 조회하여,
    가장 큰 번호 기반으로 새로운 Iteration 이름을 생성합니다.

    Returns:
        str: 자동 생성된 새로운 Iteration 이름
    """
    url = f"{AZURE_TRAINING_ENDPOINT}customvision/v3.3/training/projects/{AZURE_TRAINING_PROJECT_ID}/iterations"
    res = requests.get(url, headers=TRAIN_HEADERS)
    res.raise_for_status()
    iterations = res.json()

    max_num = 0
    for it in iterations:
        name = it["name"]
        if name.startswith("Iteration"):
            try:
                num = int(name.replace("Iteration", "").strip())
                max_num = max(max_num, num)
            except:
                continue
    return f"Iteration {max_num + 1}"


# =========================
# 🏷️ 태그 매핑 가져오기
# =========================
def get_tag_mapping():
    """
    프로젝트 내 등록된 태그 이름과 ID를 딕셔너리로 반환합니다.

    Returns:
        dict: {태그이름: 태그ID} 형태의 매핑
    """
    url = f"{AZURE_TRAINING_ENDPOINT}customvision/v3.3/training/projects/{AZURE_TRAINING_PROJECT_ID}/tags"
    res = requests.get(url, headers=TRAIN_HEADERS)
    res.raise_for_status()
    return {tag["name"]: tag["id"] for tag in res.json()}


# =========================
# 🔁 COCO → Azure 포맷 변환
# =========================
def convert_coco_to_azure_uploads(coco, tag_map):
    """
    COCO 포맷 라벨 데이터를 Azure 업로드용 포맷으로 변환합니다.
    (bounding box → normalized region)

    Args:
        coco (dict): COCO 형식 라벨 JSON
        tag_map (dict): {카테고리명: 태그ID} 매핑 정보

    Returns:
        dict: {파일명: [region, ...]} 형태의 업로드 데이터
    """
    uploads = {}
    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    for ann in coco["annotations"]:
        image_info = images[ann["image_id"]]
        file_name = image_info["file_name"]
        width, height = image_info["width"], image_info["height"]
        category_name = categories[ann["category_id"]]

        if category_name not in tag_map:
            continue

        x, y, w, h = ann["bbox"]
        region = {
            "tagId": tag_map[category_name],
            "left": x / width,
            "top": y / height,
            "width": w / width,
            "height": h / height,
        }

        if file_name not in uploads:
            uploads[file_name] = []
        uploads[file_name].append(region)

    return uploads


# =========================
# 📤 이미지 + 라벨 업로드
# =========================
def upload_to_custom_vision(uploads):
    """
    변환된 이미지-라벨 데이터들을 Azure에 업로드합니다.
    50개 단위로 나눠 전송합니다.

    Args:
        uploads (dict): {파일명: [region, ...]} 형태의 업로드 정보
    """
    batch, sent = [], 0
    for fname, regions in uploads.items():
        fpath = os.path.join(IMAGE_FOLDER, fname)
        if not os.path.exists(fpath):
            print(f"❌ {fname} 없음")
            continue
        with open(fpath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        batch.append({"name": fname, "contents": b64, "regions": regions})

        if len(batch) == 50:
            sent = send_batch(batch, sent)
            batch = []
    if batch:
        send_batch(batch, sent)


def send_batch(batch, sent):
    """
    50개 단위의 배치를 실제 업로드하는 함수

    Args:
        batch (list): 업로드할 이미지 리스트
        sent (int): 현재까지 업로드한 수

    Returns:
        int: 업로드 완료된 총 수
    """
    url = f"{AZURE_TRAINING_ENDPOINT}customvision/v3.3/training/projects/{AZURE_TRAINING_PROJECT_ID}/images/files"
    res = requests.post(url, headers=TRAIN_HEADERS, json={"images": batch}, timeout=120)
    print(f"[{sent + 1}–{sent + len(batch)}] ▶ {res.status_code}")
    try:
        print(json.dumps(res.json(), indent=2, ensure_ascii=False)[:300])
    except:
        print(res.text[:300])
    return sent + len(batch)


# =============================
# 🧠 새로운 Iteration 학습 트리거
# =============================
def train_new_iteration(iteration_name):
    """
    새로운 Iteration을 생성하고 학습을 요청합니다.
    advancedTraining 여부는 설정값에 따라 달라집니다.

    Args:
        iteration_name (str): 생성할 Iteration 이름
    """
    url = (
        f"{AZURE_TRAINING_ENDPOINT}customvision/v3.3/training/projects/{AZURE_TRAINING_PROJECT_ID}/train"
        f"?iterationName={iteration_name}&advancedTraining={'true' if USE_ADVANCED_TRAINING else 'false'}"
    )
    res = requests.post(url, headers=TRAIN_HEADERS)

    print(f"\n🧠 [Iteration 학습 요청]")
    print(f"📡 요청 URL: {url}")  # ✅ 이 줄만 추가해도 디버깅이 한결 쉬워짐

    if res.ok:
        print(f"✅ Iteration '{iteration_name}' 학습 요청 성공")
    else:
        print(f"❌ 학습 요청 실패 ({res.status_code})")
        try:
            print("🧾 응답 내용:", res.json())
        except:
            print("🧾 응답 텍스트:", res.text[:500])


# =========================
# ⏳ 학습 완료 대기
# =========================


def wait_for_training_completion(iteration_name, timeout=96 * 3600, interval=30):
    """
    학습 요청 후 일정 시간 동안 학습 완료 여부를 polling 방식으로 확인합니다.

    Args:
        iteration_name (str): 모니터링할 Iteration 이름
        timeout (int): 최대 대기 시간 (초)
        interval (int): 상태 체크 간격 (초)

    Returns:
        bool: 학습이 완료되었으면 True, 실패/취소/타임아웃이면 False
    """
    url = f"{AZURE_TRAINING_ENDPOINT}customvision/v3.3/training/projects/{AZURE_TRAINING_PROJECT_ID}/iterations"
    start_time = time.time()

    while time.time() - start_time < timeout:
        res = requests.get(url, headers=TRAIN_HEADERS)
        res.raise_for_status()
        iterations = res.json()

        for it in iterations:
            if it["name"] == iteration_name:
                status = it["status"]

                elapsed = time.time() - start_time
                hours = int(elapsed // 3600)  # 경과 ‘시’
                minutes = int((elapsed % 3600) // 60)  # 경과 ‘분’
                seconds = int(elapsed % 60)  # 경과 ‘초’

                print(
                    f"⏳ 학습 상태: {status} 경과 시간 : {hours}시간 {minutes}분 {seconds}초"
                )
                if status == "Completed":

                    print(
                        f"✅ 학습 완료! 총 소요 시간: {hours}시간 {minutes}분 {seconds}초"
                    )
                    return True
                elif status in ["Failed", "Canceled"]:
                    print(f"❌ 학습 실패 또는 취소됨: {status}")
                    return False
        time.sleep(interval)

    print("❗ 학습 완료까지 기다리다 타임아웃됨")
    return False


# =========================
# 🚀 학습 완료 후 퍼블리시
# =========================


def publish_iteration(
    iteration_name: str, publish_name: str = None, prediction_resource_id: str = None
):
    if not prediction_resource_id:
        raise ValueError("prediction_resource_id(ARM Resource ID) 가 필요합니다.")

    # 1) Iteration 목록 조회
    it_url = f"{AZURE_TRAINING_ENDPOINT}/customvision/v3.3/training/projects/{AZURE_TRAINING_PROJECT_ID}/iterations"
    it_res = requests.get(it_url, headers=TRAIN_HEADERS)
    it_res.raise_for_status()

    iteration_id = None
    for it in it_res.json():
        if it["name"] == iteration_name:
            iteration_id = it["id"]
            break

    if not iteration_id:
        raise ValueError(f"Iteration 이름 '{iteration_name}'을(를) 찾지 못했습니다.")

    # 2) 퍼블리시 요청
    publish_url = (
        f"{AZURE_TRAINING_ENDPOINT}/customvision/v3.3/training/projects/"
        f"{AZURE_TRAINING_PROJECT_ID}/iterations/{iteration_id}/publish"
        f"?predictionId={AZURE_PREDICTION_RESOURCE_ID}"
        f"&publishName={publish_name or iteration_name}"
    )

    res = requests.post(publish_url, headers=TRAIN_HEADERS)  # body 없이!
    if res.ok:
        print(f"✅ 퍼블리시 완료: {publish_name or iteration_name}")
    else:
        print(f"❌ 퍼블리시 실패: {res.status_code}\n{res.text}")
        print("📡 호출 URL:", publish_url)


# =========================
# 🚀 학습 및 퍼블리시 자동 실행
# =========================
def upload_and_train():
    # 📂 COCO 라벨 파일 로드
    with open(COCO_FILE_PATH, encoding="utf-8") as f:
        coco = json.load(f)
    print(f"📁 이미지 수: {len(coco['images'])} / 라벨 수: {len(coco['annotations'])}")

    # 🏷️ 프로젝트 내 태그 목록 조회 및 매핑
    tag_map = get_tag_mapping()

    # 🔁 COCO → Azure 업로드 포맷으로 변환
    uploads = convert_coco_to_azure_uploads(coco, tag_map)

    # 📤 이미지 + 라벨 업로드
    upload_to_custom_vision(uploads)

    # 🆕 Iteration 이름 자동 생성
    iteration_name = get_next_iteration_name()
    print(f"🚀 새 Iteration 이름 자동 설정됨: {iteration_name}")

    # 🧠 학습 트리거
    train_new_iteration(iteration_name)

    # ⏳ 학습 완료 여부 체크 → 퍼블리시 수행
    if wait_for_training_completion(iteration_name):
        publish_iteration(
            iteration_name=iteration_name,
            publish_name=iteration_name,  # 예측 시 URL에 쓰일 이름
            prediction_resource_id=AZURE_PREDICTION_RESOURCE_ID,
        )
    else:
        print("⚠️ 퍼블리시 생략: 학습이 완료되지 않았습니다.")


# =========================
# 🏁 메인 실행
# =========================
if __name__ == "__main__":
    upload_and_train()
