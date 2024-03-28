# analysis_image
# import matplotlib.pyplot as plt
from PIL import Image

def analysis_image_process(st,tab,uploaded_Image):

    for img in uploaded_Image:
        image = Image.open(img)
        with tab:
            st.image(image, caption=img.name + ' size:' + str(img.size))
