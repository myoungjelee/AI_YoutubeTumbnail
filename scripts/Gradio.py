import gradio as gr
import requests
import json
from PIL import Image, ImageDraw, ImageFont
import io
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import os
import tempfile
from dotenv import load_dotenv
import uuid
import tabulate

load_dotenv()

VIEWCOUNT_MODEL = {
    "key": os.getenv("AZURE_PREDICTION_KEY"),
    "endpoint": os.getenv("AZURE_PREDICTION_ENDPOINT"),
    "project_id": os.getenv("AZURE_PREDICTION_PROJECT_ID"),
    "published_name": "Iteration%205",
}

TRENDING_MODEL = {
    "key": os.getenv("AZURE_PREDICTION_KEY"),
    "endpoint": os.getenv("AZURE_PREDICTION_ENDPOINT"),
    "project_id": os.getenv("AZURE_PREDICTION_PROJECT_ID"),
    "published_name": "Iteration%206",
}

LABELS = ["텍스트", "브랜드/로고", "인물", "캐릭터"]

try:
    font_path = next(
        (
            f.fname
            for f in font_manager.fontManager.ttflist
            if "Malgun Gothic" in f.name
            or "AppleGothic" in f.name
            or "NanumGothic" in f.name
        ),
        None,
    )
    if font_path is None:
        raise IOError
    plt.rcParams["font.family"] = font_manager.FontProperties(
        fname=font_path
    ).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    label_font = ImageFont.truetype(font_path, 20)
except IOError:
    print("경고: 한글 폰트를 찾을 수 없습니다.")
    label_font = ImageFont.load_default()


# ✅ 약관 HTML 파일 불러오기 함수
def get_terms_html(opened=False):
    try:
        with open("terms.html", "r", encoding="utf-8") as f:
            body = f.read()

        html_id = f"terms-details-{uuid.uuid4().hex}"  # 매번 고유 ID
        if opened:
            return f"""
<details id="{html_id}" open>
  <summary>📄 이용약관 및 개인정보처리방침 (필수 동의)</summary>
  {body}
</details>
"""
        else:
            return f"""
<details id="{html_id}">
  <summary>📄 이용약관 및 개인정보처리방침 (필수 동의)</summary>
  {body}
</details>
"""
    except FileNotFoundError:
        return "<p style='color:red;'>⚠️ 약관 파일을 찾을 수 없습니다.</p>"


def predict_with_model(image, model_config):
    prediction_url = f"{model_config['endpoint']}customvision/v3.0/Prediction/{model_config['project_id']}/detect/iterations/{model_config['published_name']}/image"
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()
    headers = {
        "Prediction-Key": model_config["key"],
        "Content-Type": "application/octet-stream",
    }
    response = requests.post(prediction_url, headers=headers, data=img_byte_arr)
    response.raise_for_status()
    return response.json()


def calculate_similarity_score(predictions):
    weights = {"인물": 1.2, "텍스트": 1.1, "브랜드로고": 1.0, "캐릭터": 0.9}
    weighted_score = 0
    total_weight = 0
    for pred in predictions["predictions"]:
        weight = weights.get(pred["tagName"], 1.0)
        weighted_score += pred["probability"] * weight
        total_weight += weight
    return min((weighted_score / total_weight) * 100 if total_weight else 0, 100)


def generate_recommendations(viewcount_score, trending_score, predictions):
    recommendations = []
    if viewcount_score < 70:
        recommendations.append(
            "📈 조회수 향상 팁: 인물의 표정이나 포즈를 더 역동적으로 만들어보세요"
        )
        recommendations.append(
            "💡 텍스트 크기를 키우고 대비를 높여 가독성을 개선하세요"
        )
    if trending_score < 70:
        recommendations.append(
            "🔥 트렌드 팁: 현재 인기 있는 밝은 색상 팔레트를 사용해보세요"
        )
        recommendations.append(
            "✨ 캐릭터나 이모지 요소를 추가하면 트렌드에 더 부합할 수 있어요"
        )
    if viewcount_score > 80 and trending_score > 80:
        recommendations.append(
            "🎉 완벽해요! 현재 썸네일은 조회수와 트렌드 모두에 최적화되어 있습니다"
        )
    return recommendations


def analyze_thumbnail(image, analysis_type, threshold):
    if image is None:
        return None, "이미지를 업로드해주세요.", "", "", [], None

    viewcount_result = (
        predict_with_model(image, VIEWCOUNT_MODEL)
        if analysis_type != "트렌드 중심"
        else {"predictions": []}
    )
    trending_result = (
        predict_with_model(image, TRENDING_MODEL)
        if analysis_type != "조회수 중심"
        else {"predictions": []}
    )

    viewcount_score = calculate_similarity_score(viewcount_result)
    trending_score = calculate_similarity_score(trending_result)

    annotated_image = draw_comparison_results(
        image.copy(), viewcount_result, trending_result, threshold
    )

    result_text = f"""
🎯 **썸네일 분석 결과**
result_text = ""  # 점수 출력 라인 제거

📊 **조회수 예측 모델**: {viewcount_score:.1f}%  
🔥 **트렌드 분석 모델**: {trending_score:.1f}%

📈 **종합 점수**: {(viewcount_score + trending_score) / 2:.1f}%
"""
    result_text = ""
    detailed_analysis = create_detailed_analysis(viewcount_result, trending_result)
    recommendations = generate_recommendations(
        viewcount_score, trending_score, viewcount_result
    )
    recommendations_md = "### 💡 맞춤 개선 제안\n" + "\n".join(
        [f"- {rec}" for rec in recommendations]
    )
    file_path = generate_report_file(result_text, detailed_analysis, recommendations_md)

    return (
        annotated_image,
        result_text,  # 여기는 이제 내용 없음
        detailed_analysis,
        create_comparison_chart(viewcount_result, trending_result),
        recommendations_md,
        gr.update(visible=True),
        file_path,
        f"{(viewcount_score + trending_score) / 2:.1f}%",  # 🏆 종합 점수
        f"{viewcount_score:.1f}%",  # 👀 조회수 기여도
        f"{trending_score:.1f}%",  # 🔥 트렌드 부합도
    )


def draw_comparison_results(
    image, viewcount_result, trending_result, threshold=0.5, color_by="model"
):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    label_colors = {
        "브랜드/로고": "#FF4136",
        "인물": "#0074D9",
        "텍스트": "#FFDC00",
        "캐릭터": "#B10DC9",
    }

    model_colors = {
        "조회수": "#0074D9",  # 파랑
        "트렌드": "#FF4136",  # 빨강
    }

    def draw_prediction(pred, model_type):
        tag = pred["tagName"]
        bbox = pred["boundingBox"]
        confidence = pred["probability"]
        if confidence < threshold:
            return

        left = bbox["left"] * width
        top = bbox["top"] * height
        right = left + bbox["width"] * width
        bottom = top + bbox["height"] * height

        # ✅ 색상 선택 기준
        if color_by == "label":
            color = label_colors.get(tag, "#AAAAAA")
        else:  # color_by == "model"
            color = model_colors.get(model_type, "#AAAAAA")

        label = f"[{model_type}] {tag} {confidence:.1%}"

        # 텍스트 위치 모델에 따라 다르게
        text_xy = (
            (left + 5, top + 5) if model_type == "조회수" else (left + 5, bottom - 25)
        )

        draw.rectangle([left, top, right, bottom], outline=color, width=7)
        draw.text(text_xy, label, fill=color, font=label_font)

    for pred in viewcount_result["predictions"]:
        draw_prediction(pred, "조회수")
    for pred in trending_result["predictions"]:
        draw_prediction(pred, "트렌드")

    return image


def create_detailed_analysis(viewcount_result, trending_result):
    data = []
    for label in LABELS:
        vc_score = next(
            (
                p["probability"]
                for p in viewcount_result["predictions"]
                if p["tagName"] == label
            ),
            0,
        )
        tr_score = next(
            (
                p["probability"]
                for p in trending_result["predictions"]
                if p["tagName"] == label
            ),
            0,
        )
        data.append(
            {
                "카테고리": label,
                "조회수 모델": f"{vc_score:.1%}",
                "트렌드 모델": f"{tr_score:.1%}",
                "차이": f"{abs(vc_score - tr_score):.1%}",
            }
        )
    return pd.DataFrame(data)


def create_comparison_chart(viewcount_result, trending_result):
    categories = ["텍스트", "브랜드/로고", "캐릭터", "인물"]
    x = np.arange(len(categories))
    viewcount_scores = [
        next(
            (
                p["probability"]
                for p in viewcount_result["predictions"]
                if p["tagName"] == cat
            ),
            0,
        )
        for cat in categories
    ]
    trending_scores = [
        next(
            (
                p["probability"]
                for p in trending_result["predictions"]
                if p["tagName"] == cat
            ),
            0,
        )
        for cat in categories
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, viewcount_scores, marker="o", label="조회수 모델", color="#F08080")
    ax.plot(x, trending_scores, marker="s", label="트렌드 모델", color="#9C90EE")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_title("카테고리별 모델 비교")
    ax.grid(True)
    ax.legend()
    return fig


def generate_report_file(result_text, detailed_df, recommendations_md):
    content = (
        result_text
        + "\n\n"
        + recommendations_md
        + "\n\n"
        + detailed_df.to_markdown(index=False)
    )
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".txt", mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        return tmp.name


# ───────────────────────────────
# ✅ Gradio 인터페이스
# ───────────────────────────────

demo = gr.Blocks(title="썸네일 성공 예측기", theme=gr.themes.Soft())
with demo:
    gr.Markdown(
        "## 🚀 썸네일 성공 예측기\nAI가 당신의 썸네일 성공 가능성을 분석합니다."
    )

    current_results = gr.State(None)

    with gr.Tabs():
        with gr.TabItem("🎯 썸네일 분석"):
            terms_html = gr.HTML(
                value=get_terms_html(opened=False), elem_id="terms-html"
            )  # ⬅️ 처음엔 접혀서 시작

            agree = gr.Checkbox(label="✅ 위 약관에 동의합니다", value=False)

            agree.change(
                fn=lambda x: gr.update(value=get_terms_html(opened=False)),
                inputs=[agree],
                outputs=[terms_html],
            )

            with gr.Row(visible=False) as main_interface:
                with gr.Column():
                    image_input = gr.Image(
                        label="썸네일 업로드", type="pil", height=300
                    )
                    analysis_type = gr.Radio(
                        choices=["종합 분석", "조회수 중심", "트렌드 중심"],
                        value="종합 분석",
                        label="🔍 분석 유형",
                    )
                    confidence_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.05,
                        label="신뢰도 임계값",
                    )
                    analyze_btn = gr.Button("🚀 분석 시작", variant="primary")
                with gr.Column():
                    result_image = gr.Image(label="📊 분석 결과")
                    with gr.Row():
                        gr.Markdown(
                            "<div style='display:flex; align-items:center;'>"
                            "<div style='width:20px; height:20px; background-color:#0074D9; display:inline-block; margin-right:8px;'></div>"
                            "<span>조회수 모델</span>"
                            "</div>",
                            elem_id="legend-view",
                        )

                        gr.Markdown(
                            "<div style='display:flex; align-items:center;'>"
                            "<div style='width:20px; height:20px; background-color:#FF4136; display:inline-block; margin-right:8px;'></div>"
                            "<span>트렌드 모델</span>"
                            "</div>",
                            elem_id="legend-trend",
                        )
                    result_text = gr.Markdown(label="📝 분석 리포트")

                with gr.Group(visible=False) as detail_section:
                    gr.Markdown("### 📈 2. AI 분석 대시보드")

                    with gr.Row():
                        detailed_df = gr.Dataframe(label="카테고리별 점수")

                    with gr.Row():
                        overall_score_card = gr.Textbox(
                            label="🏆 종합 점수", interactive=False
                        )
                        view_score_card = gr.Textbox(
                            label="👀 조회수 기여도", interactive=False
                        )
                        trend_score_card = gr.Textbox(
                            label="🔥 트렌드 부합도", interactive=False
                        )

                    with gr.Row():
                        comparison_chart = gr.Plot(label="모델 비교 차트")

                    with gr.Row():
                        recommendations_text = gr.Markdown()

                    with gr.Row():
                        download_btn = gr.DownloadButton(
                            label="📥 리포트 다운로드", value=None
                        )

    def toggle_interface(agree_status):
        return gr.update(visible=agree_status), gr.update(visible=agree_status)

    def run_analysis(image, analysis_type, threshold, agree_status):
        if not agree_status:
            raise gr.Error("⚠️ 분석을 시작하려면 약관에 동의해야 합니다.")
        if image is None:
            raise gr.Error("📸 썸네일 이미지를 업로드해주세요.")
        return analyze_thumbnail(image, analysis_type, threshold)

    agree.change(
        toggle_interface, inputs=agree, outputs=[main_interface, detail_section]
    )

    analyze_btn.click(
        run_analysis,
        inputs=[image_input, analysis_type, confidence_slider, agree],
        outputs=[
            result_image,
            result_text,  # 👈 이제 이건 빈 문자열이라 화면에 안 보여짐
            detailed_df,
            comparison_chart,
            recommendations_text,
            detail_section,
            download_btn,
            overall_score_card,  # 🏆 종합 점수
            view_score_card,  # 👀 조회수 기여도
            trend_score_card,  # 🔥 트렌드 부합도
        ],
    )

demo.launch(share=True)
