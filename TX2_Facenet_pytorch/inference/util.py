from PIL import ImageDraw, ImageFont
import os


def draw_bb_on_img(faces, img):
    draw = ImageDraw.Draw(img)
    fs = max(20, round(img.size[0] * img.size[1] * 0.000005))
    font = ImageFont.truetype('fonts/NanumGothicBold.ttf', fs)
    margin = 5

    path_face_dir = '/home/tx2/Downloads/face-recognition-master/images'
    img_list = os.listdir((path_face_dir))

    for face in faces:
        if face.top_prediction.confidence * 100 >= 70:
            if face.top_prediction.label in img_list:
                text = "%s %.2f%%" % (face.top_prediction.label.upper(), face.top_prediction.confidence * 100)
                text_size = font.getsize(text)
                # bounding box
                draw.rectangle(
                    (
                        (int(face.bb.left), int(face.bb.top)),
                        (int(face.bb.right), int(face.bb.bottom))
                    ),
                    outline='green',
                    width=2
                )

                # text background
                draw.rectangle(
                    (
                        (int(face.bb.left - margin), int(face.bb.bottom) + margin),
                        (int(face.bb.left + text_size[0] + margin), int(face.bb.bottom) + text_size[1] + 3 * margin)
                    ),
                    fill='black'
                )

                # text
                draw.text(
                    (int(face.bb.left), int(face.bb.bottom) + 2 * margin),
                    text,
                    font=font
                )
        elif face.top_prediction.confidence * 100 < 70 or face.top_prediction.label not in img_list:
            text = "미등록"
            text_size = font.getsize(text)
            draw.rectangle(
                (
                    (int(face.bb.left), int(face.bb.top)),
                    (int(face.bb.right), int(face.bb.bottom))
                ),
                outline='blue',
                width=2
            )
            draw.rectangle(
                (
                    (int(face.bb.left - margin), int(face.bb.bottom) + margin),
                    (int(face.bb.left + text_size[0] + margin), int(face.bb.bottom) + text_size[1] + 3 * margin)
                ),
                fill='black'
            )
            draw.text(
                (int(face.bb.left), int(face.bb.bottom) + 2 * margin),
                text,
                font=font
            )
