# analysis_image
# import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas

def analysis_image_process(st,tab,uploaded_Image):
    # Canvas 설정
    stroke_width = 5
    stroke_color = "#ff0000"  # 붉은색

    for img in uploaded_Image:
        # image = Image.open(img)
        canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # 필러 색상, 여기서는 사용하지 않음
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_image= Image.open(img) if img else None,
                    update_streamlit=True,
                    # width=image.width,
                    # height=image.height,
                    width=300,
                    height=200,
                    drawing_mode="rect",
                    display_toolbar=True,
                    key="full_app"
                )

#    with tab:
    # if canvas_result.image_data is not None:
    #     tab.st.image(canvas_result.image_data, caption=img.name)
        

