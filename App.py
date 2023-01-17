from copyreg import pickle
from venv import create
from flask import Flask, render_template, Response, url_for, redirect, request
import cv2
import pickle
import numpy as np
import glob
import os
from PIL import Image
from datetime import datetime
import timeit

app=Flask(__name__)
cam = cv2.VideoCapture(0)
cam.set(3, 990) 
cam.set(4, 990) 

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)


names = []
face_id = 0
face_id = pickle.load(open("faceid.dat", "rb"))
names = pickle.load(open("names.dat", "rb"))
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

font = cv2.FONT_HERSHEY_SIMPLEX


#Função que marca a presença, que recebe o nome da pessoa reconhecida pelo reconhecedor
def MarcarPresença(name):
#Recebe a data atual como dia-mês-ano
    inicio = timeit.default_timer()
    now = datetime.now()
    current_date = now.strftime("%d-%m-%Y")
#Cria a lista de chamada nomeada com a data atual (dia-mês-ano)
    open(current_date+'.csv','a+')
    with open(current_date+'.csv','r+') as f:
        datalist = f.readlines()
        namelist = []
#Percorre as linhas da tabela recebendo os alunos presetes
        for line in datalist:
            entry = line.split(',')
            namelist.append(entry[0])
#Caso o aluno reconhecido não esteja na lista de chamada, e não seja um desconhecido. Ele é marcado como presente
        if name not in namelist and name != 'desconhecido':
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            fim = timeit.default_timer()
            tempo = fim - inicio
            tempos = str(tempo)
            pickle.dump(tempos, open("temporeconhecimento.txt", "wb"))

def gen():
    while True:
        def get_frame():
            ret,frame=cam.read()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.3,
                minNeighbors = 4,
            )
            for x,y,w,h in faces:
                x1,y1=x+w, y+h
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 1)
                cv2.line(frame, (x,y), (x+30, y),(255,0,255), 6) #Top Left
                cv2.line(frame, (x,y), (x, y+30),(255,0,255), 6)

                cv2.line(frame, (x1,y), (x1-30, y),(255,0,255), 6) #Top Right
                cv2.line(frame, (x1,y), (x1, y+30),(255,0,255), 6)

                cv2.line(frame, (x,y1), (x+30, y1),(255,0,255), 6) #Bottom Left
                cv2.line(frame, (x,y1), (x, y1-30),(255,0,255), 6)

                cv2.line(frame, (x1,y1), (x1-30, y1),(255,0,255), 6) #Bottom right
                cv2.line(frame, (x1,y1), (x1, y1-30),(255,0,255), 6)
                cv2.putText(frame, str("Detectado"), (x+5,y-5), font, 1, (255,255,255), 2)
            ret,jpg=cv2.imencode('.jpg',frame)
            return jpg.tobytes()
        frame=get_frame()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')



#Função que realiza a captura de video e o reconhecimento facial
def gen_frames():
#A lista de alunos cadastrados é carregada
    names = pickle.load(open("names.dat", "rb"))
#Realiza a leitura dos frames capturados
    while True:

        ret, frame = cam.read()
   
#Converte os frames para preto e branco, e define os parametros do cascade 
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.05,
            minNeighbors = 4,
            minSize = (int(minW), int(minH)),
        )
#Realiza a detecção e reconhecimento facial
        for(x,y,w,h) in faces:
#Desenha um retangulo envolta do rosto detectado
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
#realiza o reconhecimento facial
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

#Checa se a confiança (confidence) é menor que 100, sendo que "0" é uma combinação perfeita 
            if (confidence < 100):
                Id = names[id] 
                confidence = "  {0}%".format(round(100 - confidence))
#Caso o rosto não seja reconhecido, ele é rotulado como desconhecido 
            else:
                Id = "desconhecido"
                confidence = "  {0}%".format(round(100 - confidence))
#A função de marcar presença é chamada informando o nome do aluno
            MarcarPresença(Id)
#A imagem é transmitida na tela, juntamente com o rosto enquadrado com o retangulo, % de confiança e o nome do aluno
            cv2.putText(frame, str(Id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
            
   
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')   

#Função que converte as imagens em dados numéricos para treinamento do algoritmo
def getImagesAndLabels():
    path = 'dataset'
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
#Percorre todas as imagens do diretório e as converte em arrays
    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

#Função que captura as imagens para cadastrar um aluno
def dataset(nome):
    count = 0
    while True:
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        
        face_id = pickle.load(open("faceid.dat", "rb"))
        names = pickle.load(open("names.dat", "rb"))
        
        
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1

#Salva a imagem capturada no diretório selecionado 
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

#Caso as 100 imagens do aluno ja tenham sido capturadas, a função é finalizada
        if count >= 100: 
            names.append(nome)
            face_id += 1
            pickle.dump(names, open("names.dat", "wb"))
            pickle.dump(face_id, open("faceid.dat", "wb"))
            return render_template('Cconfig.html')       
            break

#Reseta o sistema completamente
def limpeza():
    names.clear()
    face_id = 0
    pickle.dump(names, open("names.dat", "wb"))
    pickle.dump(face_id, open("faceid.dat", "wb"))
    dir = 'dataset'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    return render_template('home.html')

#Realiza o treino do sistema
def treino():
    inicio = timeit.default_timer()
    faces,ids = getImagesAndLabels()
    recognizer.train(faces, np.array(ids))
#Salva os dados de treinamento no diretório estabelecido
    recognizer.write('trainer/trainer.yml')
    fim = timeit.default_timer()
    tempo = fim - inicio
    tempos = str(tempo)
    pickle.dump(tempos, open("tempotreino.txt", "wb"))

    return render_template('home.html')           

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/cadastro', methods = ["POST", "GET"])
def cadastro():
    if request.method == "POST":
        user = request.form["nome"]
        return Response(dataset(user))
    else:
        return render_template('cadastro.html')

@app.route('/config', methods = ["POST", "GET"])
def config():
    if request.method == "POST":
        return Response(limpeza())
    else:
        return render_template('config.html')

@app.route('/cadastro/config')
def Cconfig():
    return render_template('Cconfig.html')

@app.route('/cadastro/treinar', methods = ["POST", "GET"])
def treinar():
    if request.method == "POST":
        return Response(treino())
    else:
        return render_template('treinar.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect')
def detect():
    return Response(gen(),
    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/chamada')
def chamada():
    return render_template('chamada.html')

if __name__=='__main__':
    app.run(debug=True)