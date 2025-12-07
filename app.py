import os
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, render_template
import base64

app = Flask(__name__)

# Configuração
UPLOAD_FOLDER = '/tmp/uploads'
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except:
    pass

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- FUNÇÕES ---
def calculate_angle(a,b,c):
    a = np.array(a);b = np.array(b);c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def calculate_distance(a, b):
    a = np.array(a); b = np.array(b)
    return np.linalg.norm(a - b)

def analyze_stance(landmarks):
    try:
        sh_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        sh_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        an_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        an_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        shoulder_width = calculate_distance(sh_l, sh_r)
        feet_width = calculate_distance(an_l, an_r)
        ratio = feet_width / shoulder_width
        if 0.9 < ratio < 1.5: return ("Posição Inicial: ✅ Bom Equilíbrio!", "Seus pés estão na largura dos ombros.")
        else: return ("Posição Inicial: ⚠️ Ponto de Melhoria.", "Seus pés parecem estar muito juntos ou afastados.")
    except: return None

def analyze_trophy_pose(elbow_angle, shoulder_angle):
    if elbow_angle < 100 and shoulder_angle > 80: return ("Posição de Troféu: ✅ Ótima Posição!", "Você atinge uma boa posição de troféu.")
    else: return (f"Posição de Troféu: ⚠️ Ponto de Melhoria.", f"Ângulo do cotovelo de {int(elbow_angle)}°, ombro de {int(shoulder_angle)}°.")

def analyze_contact(max_arm_angle):
    if max_arm_angle >= 165: return (f"Ponto de Contato: ✅ Excelente Extensão!", f"Extensão de {int(max_arm_angle)} graus.")
    else: return (f"Ponto de Contato: ⚠️ Ponto de Melhoria.", f"Extensão de apenas {int(max_arm_angle)}°.")

def analyze_follow_through(wrist_final_pos, opposite_hip_pos):
    if wrist_final_pos is None or opposite_hip_pos is None: return None
    if wrist_final_pos[0] < opposite_hip_pos[0]: return ("Terminação: ✅ Bom Movimento!", "Você completa o movimento cruzando o corpo.")
    else: return ("Terminação: ⚠️ Ponto de Melhoria.", "Sua terminação parece curta.")

def process_video_for_image_and_feedback(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, [("Erro", "Erro ao abrir vídeo.")]

    feedback_list = []
    initial_landmarks, min_elbow_angle, trophy_shoulder_angle, max_arm_angle, final_wrist_pos, final_hip_pos = (None, 180, 0, 0, None, None)
    best_frame_index = -1
    current_frame_index = 0

    # A CORREÇÃO ESTÁ AQUI: O 'with' garante que a IA ligue e desligue corretamente
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Otimização de tamanho
            if frame.shape[1] > 1000:
                scale = 1000 / frame.shape[1]
                frame = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                if initial_landmarks is None: initial_landmarks = landmarks
                try:
                    shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]; elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]; wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]; hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]; hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                    arm_angle = calculate_angle(shoulder_r, elbow_r, wrist_r)
                    if arm_angle > max_arm_angle:
                        max_arm_angle = arm_angle
                        best_frame_index = current_frame_index

                    shoulder_angle_val = calculate_angle(hip_r, shoulder_r, elbow_r)
                    if arm_angle < min_elbow_angle: min_elbow_angle = arm_angle; trophy_shoulder_angle = shoulder_angle_val
                    final_wrist_pos = wrist_r; final_hip_pos = hip_l
                except: pass
            current_frame_index += 1

    image_base64 = None
    if best_frame_index != -1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_index)
        ret, frame = cap.read()
        if ret:
            if frame.shape[1] > 1000:
                scale = 1000 / frame.shape[1]
                frame = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))

            # Recria a IA apenas para desenhar na foto final
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_draw:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose_draw.process(image_rgb)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                _, buffer = cv2.imencode('.jpg', frame)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
    cap.release()

    if initial_landmarks: feedback_list.append(analyze_stance(initial_landmarks))
    feedback_list.append(("Lançamento (Toss): 🚧 Em Construção.", "A detecção da bola virá em breve."))
    if min_elbow_angle < 170: feedback_list.append(analyze_trophy_pose(min_elbow_angle, trophy_shoulder_angle))
    if max_arm_angle > 0: feedback_list.append(analyze_contact(max_arm_angle))
    if final_wrist_pos and final_hip_pos: feedback_list.append(analyze_follow_through(final_wrist_pos, final_hip_pos))

    return image_base64, feedback_list

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '': return render_template('index.html', results=[("Erro", "Envie um arquivo.")])

        # Cria pasta se não existir
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        image_data, results_list = process_video_for_image_and_feedback(filepath)

        try: os.remove(filepath)
        except: pass
        return render_template('index.html', results=results_list, image_base64=image_data)
    return render_template('index.html', results=None, image_base64=None)

if __name__ == "__main__":
    # Garante que roda na porta certa (Render ou Local)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)