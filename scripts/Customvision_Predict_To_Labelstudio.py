"""
📌 목적: Azure Custom Vision에서 최신 퍼블리시된 Iteration을 기준으로
        로컬 이미지에 대해 객체 감지 예측을 수행하고,
        결과를 COCO 및 Label Studio 포맷으로 변환한 뒤,
        Label Studio를 자동 실행하는 전체 자동화 파이프라인 스크립트

🧩 주요 기능 및 워크플로우:
1. 🖼️ 로컬 이미지 폴더에서 이미지 목록 수집
2. 🏷️ Custom Vision 프로젝트 이름 조회
3. 🔁 가장 최근 퍼블리시된 Iteration의 예측 URL 자동 조회
4. 🧠 해당 Iteration을 통해 각 이미지에 대해 객체 감지 예측 수행
5. 📦 예측 결과를 기반으로:
   - COCO 포맷 (`images`, `annotations`, `categories`)으로 변환
   - Label Studio 포맷으로 변환 (`/data/local-files/?d=...` 포함)
6. 💾 JSON 파일로 저장 (`azure_to_labelstudio.json`)
7. 🛑 기존 Label Studio 인스턴스 종료 (PID 기반)
8. 🚀 환경변수 기반으로 Label Studio 자동 실행

🔧 사전 준비사항:
- 프로젝트 ID(나머진 다 같은 리소스 그룹이라 동일)
- IMAGE_FOLDER도 올리려는 이미지 경로에 맞게 수정
"""

import os, json, subprocess, signal, requests
from PIL import Image
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
# =========================
# ⚙️ 설정값(변경해야할 값)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "data", "Youtube_Tranding", "thumbnails2")

LABEL_INFO = {
    "브랜드/로고": {"id": 1, "threshold": 0.9},
    "인물": {"id": 2, "threshold": 0.9},
    "캐릭터": {"id": 3, "threshold": 0.9},
    "텍스트": {"id": 4, "threshold": 0.9},
}

# =========================
# 🔧 설정값(기본 고정값)
# =========================
AZURE_PREDICTION_PROJECT_ID = os.getenv("AZURE_PREDICTION_PROJECT_ID")  # CV프로젝트 ID
AZURE_TRAINING_KEY = os.getenv("AZURE_TRAINING_KEY")
AZURE_PREDICTION_KEY = os.getenv("AZURE_PREDICTION_KEY")
AZURE_TRAINING_ENDPOINT = os.getenv("AZURE_TRAINING_ENDPOINT")
AZURE_PREDICTION_ENDPOINT = os.getenv("AZURE_PREDICTION_ENDPOINT")

OUTPUT_PATH = os.path.join(BASE_DIR, "azure_to_labelstudio.json")

LABEL_STUDIO_PID = "labelstudio.pid"


# =================================
# 🏷️ Custom Vision 프로젝트 이름 조회
# ================================
def get_customvision_project_name(
    project_id: str, endpoint: str, training_key: str
) -> str:
    headers = {"Training-Key": training_key}
    url = f"{endpoint}customvision/v3.3/training/projects/{project_id}"
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    return res.json()["name"]


# ===============================
# 🔁 퍼블리시된 Iteration 정보 조회
# ==============================
def get_latest_published_iteration_url(
    project_id: str, endpoint: str, training_key: str
) -> str:
    """
    Azure Custom Vision에서 가장 최근에 퍼블리시된 Iteration의 prediction URL을 반환합니다.

    Args:
        project_id (str): Custom Vision 프로젝트 ID
        endpoint (str): Azure 서비스 엔드포인트 URL
        training_key (str): Training API 키

    Returns:
        str: 최신 퍼블리시 Iteration의 prediction URL

    Raises:
        RuntimeError: 퍼블리시된 Iteration이 없을 경우 예외 발생
    """
    training_endpoint = endpoint.replace("-prediction", "")
    headers = {"Training-Key": training_key}
    url = f"{training_endpoint}customvision/v3.3/training/projects/{project_id}/iterations"
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    iterations = res.json()

    published = [it for it in iterations if it.get("publishName")]
    if not published:
        raise RuntimeError("퍼블리시된 Iteration이 없습니다.")

    latest = sorted(published, key=lambda it: it["created"], reverse=True)[0]
    iteration_name = latest["publishName"]
    project_name = get_customvision_project_name(
        AZURE_PREDICTION_PROJECT_ID, AZURE_TRAINING_ENDPOINT, AZURE_TRAINING_KEY
    )
    print(
        f"🔁 예측 대상 프로젝트: '{project_name}' / 사용된 Iteration: '{iteration_name}'"
    )

    return f"{AZURE_PREDICTION_ENDPOINT}customvision/v3.0/Prediction/{project_id}/detect/iterations/{iteration_name}/image"


# =========================
# 🧠 이미지 예측 수행
# =========================
def predict_image(image_path: str, prediction_url: str) -> dict:
    """
    지정된 이미지에 대해 Azure Prediction API를 호출하여 예측 결과를 반환합니다.

    Args:
        image_path (str): 로컬 이미지 경로
        prediction_url (str): 예측용 Iteration URL

    Returns:
        dict: 예측 결과 JSON
    """
    with open(image_path, "rb") as f:
        response = requests.post(
            prediction_url,
            headers={
                "Prediction-Key": AZURE_PREDICTION_KEY,
                "Content-Type": "application/octet-stream",
            },
            data=f,
        )
    response.raise_for_status()  # 200~299가 아니면 여기서 에러 발생!
    return response.json()


# =========================
# 🧾 COCO 포맷 변환
# =========================
def convert_to_coco(images: List[str], prediction_url: str) -> dict:
    """
    이미지 리스트와 예측 결과를 기반으로 COCO 포맷으로 변환합니다.

    Args:
        images (List[str]): 이미지 경로 리스트
        prediction_url (str): 예측 요청 URL

    Returns:
        dict: COCO 포맷의 딕셔너리 (images, annotations, categories 포함)
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": v["id"], "name": k} for k, v in LABEL_INFO.items()],
    }
    ann_id = 1

    for img_id, img_path in enumerate(images, 1):
        with Image.open(img_path) as img:
            width, height = img.size
        file_name = os.path.basename(img_path)

        coco["images"].append(
            {"id": img_id, "file_name": file_name, "width": width, "height": height}
        )

        prediction = predict_image(img_path, prediction_url)
        for pred in prediction["predictions"]:
            label = pred["tagName"]
            prob = pred["probability"]
            if label not in LABEL_INFO or prob < LABEL_INFO[label]["threshold"]:
                continue

            bbox = pred["boundingBox"]
            x, y, w, h = (
                bbox["left"] * width,
                bbox["top"] * height,
                bbox["width"] * width,
                bbox["height"] * height,
            )

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": LABEL_INFO[label]["id"],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "score": prob,
                }
            )
            ann_id += 1

    return coco


# =========================
# 🧾 Label Studio 포맷 변환
# =========================
def convert_to_labelstudio(coco: dict) -> List[dict]:
    """
    COCO 포맷을 Label Studio JSON 포맷으로 변환합니다.

    Args:
        coco (dict): COCO 포맷 딕셔너리

    Returns:
        List[dict]: Label Studio에서 사용하는 태스크 리스트 JSON
    """
    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    task_map: Dict[str, dict] = {}

    for ann in coco["annotations"]:
        image_info = images[ann["image_id"]]
        file_path = (
            f"/data/local-files/?d={IMAGE_FOLDER}/{image_info['file_name']}".replace(
                "\\", "/"
            )
        )

        if file_path not in task_map:
            task_map[file_path] = {
                "data": {"image": file_path},
                "annotations": [{"result": []}],
            }

        x, y, w, h = ann["bbox"]
        result = {
            "original_width": image_info["width"],
            "original_height": image_info["height"],
            "image_rotation": 0,
            "value": {
                "x": round(x / image_info["width"] * 100, 2),
                "y": round(y / image_info["height"] * 100, 2),
                "width": round(w / image_info["width"] * 100, 2),
                "height": round(h / image_info["height"] * 100, 2),
                "rotation": 0,
                "rectanglelabels": [categories[ann["category_id"]]],
            },
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
        }
        task_map[file_path]["annotations"][0]["result"].append(result)

    return list(task_map.values())


# =========================
# 🛑 기존 Label Studio 종료
# =========================
def stop_label_studio():
    """
    이전에 실행된 Label Studio 프로세스를 종료합니다 (PID 파일 기준).
    """
    if not os.path.exists(LABEL_STUDIO_PID):
        return

    with open(LABEL_STUDIO_PID, "r") as f:
        pid = int(f.read().strip())

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"🛑 기존 Label Studio 종료됨 (PID: {pid})")
    except ProcessLookupError:
        print(f"⚠️ 이미 종료된 PID: {pid}")
    except Exception as e:
        print(f"❌ 종료 실패: {e}")
    finally:
        os.remove(LABEL_STUDIO_PID)


# =========================
# 🚀 Label Studio 실행
# =========================
def run_label_studio():
    """
    환경 변수 설정 후, 새로운 Label Studio 인스턴스를 실행합니다.
    """
    stop_label_studio()

    env = os.environ.copy()
    env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = BASE_DIR

    process = subprocess.Popen(
        ["label-studio", "start"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    with open(LABEL_STUDIO_PID, "w") as f:
        f.write(str(process.pid))

    print(f"🚀 Label Studio 실행됨 (PID: {process.pid})")


# ===========================
# 📦 예측 수행 및 라벨 포맷 변환
# ===========================
def predict_to_labelstudio():
    # 🔍 로컬 이미지 목록 수집
    images = [
        os.path.join(IMAGE_FOLDER, f)
        for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"📁 총 이미지 개수: {len(images)}")

    # 🛰️ Azure에서 최신 퍼블리시 Iteration URL 자동 조회
    prediction_url = get_latest_published_iteration_url(
        project_id=AZURE_PREDICTION_PROJECT_ID,
        endpoint=AZURE_TRAINING_ENDPOINT,
        training_key=AZURE_TRAINING_KEY,
    )

    # 🧠 예측 수행 후 COCO 포맷으로 변환
    coco = convert_to_coco(images, prediction_url)

    # 🔁 COCO → Label Studio 포맷 변환
    label_data = convert_to_labelstudio(coco)

    # 💾 변환된 결과를 JSON 파일로 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(label_data, f, indent=2, ensure_ascii=False)

    print("✅ 변환 완료: COCO & LabelStudio JSON")


if __name__ == "__main__":
    predict_to_labelstudio()

    # 🚀 Label Studio 자동 실행
    run_label_studio()
