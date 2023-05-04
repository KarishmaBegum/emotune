import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
import googleapiclient.discovery


model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("EmoTune: A Real-Time Music Recommender Based on Facial Emotion Recognition")

if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not (emotion):
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"


class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        ##############################
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]

            print(pred)
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1,
                                                                         circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        ##############################

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


lang = st.text_input("Enter your preferred language")
singer = st.text_input("Enter your preferred singer")

if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

btn = st.button("Recommend songs")

if btn:
    if not (emotion):
        st.warning("Let's capture your emotion")
        st.session_state["run"] = "true"
    else:
        def search_music_videos(query):
            youtube = googleapiclient.discovery.build("youtube", "v3",
                                                      developerKey="AIzaSyDSa4iEvFl8cAXXf7fhbM1IjKMjt9VuXI0")
            request = youtube.search().list(
                part='snippet',  # specify the resource properties to retrieve
                q=query,  # the search query
                type='video',  # limit the search to videos
                videoDuration="medium",
                maxResults=10,  # return up to 10 results
            )
            response = request.execute()
            return response


        query = f"{lang} {emotion} song {singer}"
        results = search_music_videos(query)
        items = results['items']
        if len(items) > 0:
            table_data = []
            for item in items:
                video_url = f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                st.markdown(f"*{item['snippet']['title']}*")
                st.video(video_url)
                table_data.append([item['snippet']['title'], item['snippet']['channelTitle'], video_url])
            st.table(table_data)
        else:
            st.warning("No songs found.")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"