import tkinter as tkt
from tkinter import Message, Text, simpledialog, messagebox
import cv2,csv,os
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import pyttsx3
from twilio.rest import Client
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
import pygame
import threading

absolute_path = os.path.dirname(__file__)

window = tkt.Tk()
window.title("WISE WELLNESS")
window.configure(background='white')
window.geometry("1310x1200")
window.resizable(True, True)
message_label = tkt.Label(window, text="", bg="white", fg="green", font=('times', 15, 'bold'))
message_label.place(x=500, y=650)

image3 = Image.open(os.path.join(absolute_path, "images/bg1.png"))
image4 = image3.resize((1300,250))
photo2 = ImageTk.PhotoImage(image4)
label2 = tkt.Label(window, image = photo2)
label2.image4 = photo2
label2.place(x=0,y=25)

image = Image.open(os.path.join(absolute_path,"images/Wise_wellness_logo.png"))
image2 = image.resize((300,250))
photo = ImageTk.PhotoImage(image2)
label = tkt.Label(window, image = photo)
label.image = photo
label.place(x=650,y=152, anchor = tkt.CENTER)

#global prev_Id
prev_Id=0

# to check if text is number or not 
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

#function to take name, id and 
#sample images which is used 
# for training the model. It takes 20 images sample
# for each user

harpath = os.path.join(absolute_path,"haarcascade_frontalface_default.xml")
global path
path2 = os.path.join(absolute_path,"dataset")
#"C:\\Users\\Kaushika\\FacialRecognitionProject\\dataset\\"
usercsv = os.path.join(absolute_path,"UserDetails/UserDetails.csv")
#"C:\\Users\\Kaushika\\FacialRecognitionProject\\UserDetails\\UserDetails.csv"
trainer = os.path.join(absolute_path,"trainer/trainer.yml")
#"C:\\Users\\Kaushika\\FacialRecognitionProject\\trainer\\trainer.yml"

def TakeImages():
    
    # name and Id will be used for recognising the person
    global Id
    Id = (txt.get())
    global name
    name = (txt2.get()).lower()
    des = (txt3.get())

    # to check if id is number or not
    if(is_number(Id) and name.isalpha()):
        # opening the primary camera in laptop
        # 0 indicates primary camera    
        cam = cv2.VideoCapture(0)
        # path to haarcascade file
        # this file contains object detection algorithm
        # to identify face in real time video
        harcascadePath = harpath        
        # Creating the haarcascade classifier
        detector = cv2.CascadeClassifier(harcascadePath)
        # initial number of sample images= 0
        sampleNum = 0
        while(True):
            # Reading the video captured by camera frame by frame
            ret, img = cam.read()

            # It converts the images in different sizes
            # (decreases by 1.3 times) and 5 specifies the
            # number of times scaling happens
            # img indicates the colour of image to be considered
            faces = detector.detectMultiScale(img, 1.3, 5)
 
            # For creating a rectangle around the face
            for (x, y, w, h) in faces:
                # Specifying the coordinates of the image
                # and color and thickness of the rectangle.
                # incrementing sample number for each image
                cv2.rectangle(img, (x, y), (
                    x + w, y + h), (255, 0, 0), 2)
                sampleNum = sampleNum + 1
                # saving the captured faces in dataset folder
                cv2.imwrite(
                    path2+"\\" + name + "."+Id + '.' + str(
                        sampleNum) + ".jpg", img[y:y + h, x:x + w])
                # display the frame that has been captured
                # and drawn rectangle around it.
                cv2.imshow('frame', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is more than 19
            elif sampleNum > 19:
                break
        # releasing the camera
        cam.release()
        # closing the window
        cv2.destroyAllWindows()
        # Displaying Message 
        res = "Images Saved for ID : " + Id + " Name : " + name
        # Creating the entry of the user in a csv file
        global row
        row = [Id, name, des]
        with open(usercsv, 'a+') as csvFile:
            writer = csv.writer(csvFile)
            # Entry of the row in csv file
            writer.writerow(row)
            message_label.config(text=res)
    else:
        if(is_number(Id)):
            messagebox.showerror("Error", "Please enter alphabetical name.")
        if(name.isalpha()):
             messagebox.showerror("Error", "Please enter numeric id.")

def getImagesAndLabels(path2):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path2, f) for f in os.listdir(path2)]
    faces = []
    # creating empty ID list
    Ids = []
    # looping through all image path 
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        #converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extracting face from training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

# Training images saved in training folder
def TrainImages():
    # algorithm inside OpenCV module used for training the image dataset
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # path for HaarCascade file
    harcascadePath = harpath
    # creating detector for faces
    detector = cv2.CascadeClassifier(harcascadePath)
    # Saving the detected faces in variables
    faces, Id = getImagesAndLabels(path2)
    # Saving the trained faces and their ids
    # in file named trainer.yml
    recognizer.train(faces, np.array(Id))
    recognizer.save(trainer)
    # Displaying the Message to user
    res = "Image Trained"
    Message.configure(text=res)
 
# function to speak a text
def speak():   
    # initiating pyttsx3    
    engine = pyttsx3.init()  
    # convert this text to speech 
    # 100 is the speed to play
    engine.setProperty("rate", 100) 
    engine.say("The person is:" +aa+" "+dd)  
    # play the speech until over 
    engine.runAndWait()  

# function to recognize the face
def  TrackImages():
    global prev_Id, Id,aa,dd
    # opening csv file in which id, name and description is stored              
    f = open(usercsv, 'r')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # Reading the trained model
    recognizer.read(trainer)
    harcascadePath = harpath
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    # reading csv   
    df = pd.read_csv(f)
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 640) # set video height
    # Define min window size to be recognized as a face 
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(im, 1.2, 5)
        for(x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            # this returns the prediction of confidence of faces detected
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

            if(conf < 100): 
                dd = df.loc[df['Id'] == Id]['Desc'].values
                aa = df.loc[df['Id'] == Id]['Name'].values
            else:
                aa = 'Unknown'
            
            # recognizes and puts the name below the rectangle around face
            cv2.putText(im ,str(aa), (x, y + h),
                        font, 1, (255, 255, 255), 2)
            
            if(prev_Id != Id):
                if(Id != 'Unknown'):
                    prev_Id = Id
                    # speaks name and description of the person
                    aa = str(aa)
                    #speak(f"The person is:{aa}:{dd}")
                    threading.Thread(target=speak).start()

        cv2.imshow('im', im)
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        # if esc key is pressed, the loop will break and recognition will be over
        if k == 27:
            break
    cv2.destroyAllWindows()

# function to record voice
def record():  
    # gets name of person
    name = (txt2.get()).lower()
    if name:
        name2 = name.lower()
    else:
        messagebox.showerror("Error", "Please enter name.")

    global filename
    # path to audio file, stored with name entered
    filename = (os.path.join(absolute_path,fr"audio/{name2}.wav"))
    #"C:\Users\Kaushika\FacialRecognitionProject\audio\{name2}.wav"
    fs = 44100  # Sample rate
    seconds = 10  # records for 10 seconds
    # in-built syntax from sounddevice library to record
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write(filename, fs, myrecording)  # Save the file in the path mentioned 

# to play the audio
def play():    
    # asks the user to enter name of the person they want to here voice of  
    name = simpledialog.askstring("Play audio", "Enter the name of the person:")

    filename = os.path.join(absolute_path,fr"audio/{name}.wav")
    #fr"C:\Users\Kaushika\FacialRecognitionProject\audio\{name}.wav"
    print("playing...")
    data, fs = sf.read(filename, dtype='float32')  
    sd.play(data, fs)
    status = sd.wait()  # Wait until audio is done playing

# function with gui of face_recognition window
def face():
    global newWindow
    newWindow = tkt.Toplevel(window)
    newWindow.title("Face Recognizer")
    newWindow.geometry("1300x1200")

    
    topmessage2 = tkt.Label(newWindow, text = "FACE ASSIST",width = 40,
                height = 3, font=('times', 40, 'bold'),fg="#00688B")
    topmessage2.place(x=400, y=30)
  

    frame = tkt.Frame(newWindow)
    frame.pack()
    global img
    img= tkt.PhotoImage(file=os.path.join(absolute_path,"images/blue_background.png"), master= newWindow)
    img_label= tkt.Label(newWindow,image=img)
    img_label.place(x=0, y=0)
     
    global photo
    image = Image.open(os.path.join(absolute_path,"images/FACE ASSIST.png"))
    image2 = image.resize((200,200))
    photo = ImageTk.PhotoImage(image2)
    label = tkt.Label(newWindow, image = photo)
    label.place(x=550,y=10)


    lbl = tkt.Label(newWindow, text="ID",
                    width=15, height=2, bg="#00688B",
                    fg="white", font=('times', 20, ' bold '), relief = "raised")
    lbl.place(x=600, y=300)

    global txt
    txt = tkt.Entry(newWindow,
                    width=20, fg="#00688B",
                    bg="white", font=('times', 25, ' bold '), relief = "raised")
    txt.place(x=900, y=313)
        
    lbl2 = tkt.Label(newWindow, text="Name",
                        width=15, bg="#00688B", fg="white",
                        height=2, font=('times', 20, ' bold '),relief = "raised")
    lbl2.place(x=600, y=400)
    global txt2
    txt2 = tkt.Entry(newWindow, width=20,
                        fg="#00688B", bg="white",
                        font=('times', 25, ' bold '),relief = "raised")
    txt2.place(x=900, y=413)


    lbl3 = tkt.Label(newWindow, text="Description",
                        width=15, bg="#00688B", fg="white",
                        height=2, font=('times', 20, ' bold '),relief = "raised")
    lbl3.place(x=600, y=500)

    global txt3
    txt3 = tkt.Entry(newWindow, width=20,
                        fg="#00688B", bg="white",
                        font=('times', 25, ' bold '),relief = "raised")
    txt3.place(x=900, y=513)
        
    takeImg = tkt.Button(newWindow, text="Sample", command = TakeImages,
                     bg="white", fg="#00688B",
                        width=15, height=2, activebackground="Red",
                        font=('times', 15, ' bold '))
    takeImg.place(x=150, y=120)
    trainImg = tkt.Button(newWindow, text="Train Model", command = TrainImages,
                         bg="white", fg="#00688B",
                        width=15, height=2, activebackground="Red",
                        font=('times', 15, ' bold '))
    trainImg.place(x=150, y=220)
    trackImg = tkt.Button(newWindow, text="Face Recognize", command= TrackImages,
                         bg="white", fg="#00688B",
                        width=15, height=2, activebackground="Red",
                        font=('times', 15, ' bold '))
    trackImg.place(x=150, y=320)
    recording = tkt.Button(newWindow, text="Record voice", command=record,
                        bg="white", fg="#00688B",
                        width=15, height=2, activebackground="Red",
                        font=('times', 15, ' bold '))
    recording.place(x=150, y=420)


    playing = tkt.Button(newWindow, text="Play Audio", command = play,
                         bg="white", fg="#00688B",
                        width=15, height=2, activebackground="Red",
                        font=('times', 15, ' bold '))
    playing.place(x=150, y=520)

    quitWindow = tkt.Button(newWindow, text="Quit",
                        command=newWindow.destroy, bg="white", fg="#00688B",
                        width=15, height=2, activebackground="Red",
                        font=('times', 15, ' bold '))
    quitWindow.place(x=150, y=620)
    
# function with gui of sos window
def sos():

    global contacts    
        
    # Initialize tkinter
    main_window = tkt.Toplevel(window)
    main_window.title("SOS")
    main_window.geometry("1300x1200")


    frame2 = tkt.Frame(main_window)
    frame2.pack()
    # Load and display background image
    background_photo = tkt.PhotoImage(file = os.path.join(absolute_path,"images/blue_background.png"), master= main_window)
    background_label = tkt.Label(main_window, image=background_photo)
    background_label.place(x=0, y=0)

    # Set up other GUI elements
    topmessage2 = tkt.Label(main_window, text="SOS", width=10, height=3, font=('times',52, 'bold'), fg="#00688B")
    topmessage2.place(x=700, y=25)


    # Twilio credentials are not included in this repository.
    # Add your own credentials locally if Twilio functionality is required.

    account_sid = ''
    auth_token = ''
    twilio_phone_number = ''


    # Initialize Twilio client
    client = Client(account_sid, auth_token)

    # File to store contacts
    contacts_file = "contacts.csv"

    # Load contacts from CSV file
    def load_contacts():
        contacts = {}
        try:
            file = open(contacts_file, mode='r')
            reader = csv.reader(file)
            for row in reader:
                contacts[row[0]] = row[1]
            file.close()
        except FileNotFoundError:
            pass  # If file not found, return an empty contacts dictionary
        return contacts

    # Save contacts to CSV file
    def save_contacts():
        with open(contacts_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            for name, number in contacts.items():
                writer.writerow([name, number])

    # Load emergency message from CSV file
    def load_emergency_message():
        try:
            with open("emergency_message.txt", "r") as file:
                return file.read().strip()
        except FileNotFoundError:
            return ""

    # Save emergency message to CSV file
    def save_emergency_message(message):
        with open("emergency_message.txt", "w") as file:
            file.write(message)

    # Dictionary to store contacts
    global contacts
    contacts = load_contacts()

    # Load emergency message
    global emergency_message
    emergency_message = load_emergency_message()

    # Function to send SOS to all contacts
    def send_sos():
        if not contacts:
            messagebox.showerror("Error", "Please add contacts for SOS to send.")
            return
        if not emergency_message:
            messagebox.showerror("Error", "Please set an emergency message.")
            return
        for name, number in contacts.items():
            send_sms(number, emergency_message)
            make_call(number, emergency_message)
        messagebox.showinfo("SOS Sent", "SOS message and call sent successfully!")

    # Function to send SMS
    def send_sms(number, message):
        message = client.messages.create(
            body=message,
            from_=twilio_phone_number,
            to=number
        )
        print(f"SMS sent successfully to {number}!")

    # Function to make a call
    def make_call(number, message):
        call = client.calls.create(
            twiml=f'<Response><Say>{message}</Say></Response>',
            from_=twilio_phone_number,
            to=number
        )
        print(f"Call initiated successfully to {number}!")

    # Function to add a contact
    def add_contact():
        name = name_entry.get()
        number = number_entry.get()
        if name and number:
            if validate_phone_number(number):
                contacts[name] = number
                save_contacts()  # Save contacts to CSV
                messagebox.showinfo("Contact Added", f"Contact '{name}' with number '{number}' added successfully!")
                name_entry.delete(0, tkt.END)
                number_entry.delete(0, tkt.END)
            else:
                messagebox.showerror("Error", "Invalid phone number format. Please enter a valid number with country code.")
        else:
            messagebox.showerror("Error", "Please enter both name and number.")

    # Function to delete a contact
    def delete_contact():
        name_to_delete = simpledialog.askstring("Delete Contact", "Enter the name of the contact to delete:")
        if name_to_delete in contacts:
            del contacts[name_to_delete]
            save_contacts()
            messagebox.showinfo("Contact Deleted", f"Contact '{name_to_delete}' deleted successfully!")
        else:
            messagebox.showerror("Error", f"Contact '{name_to_delete}' not found.")

    # Function to validate phone number format (check for country code)
    def validate_phone_number(number):
        # For simplicity, assume any 10-digit number with a '+' prefix is valid
        return len(number) >= 11 and number[0] == '+' and number[1:].isdigit()

    # Function to set emergency message
    def set_emergency_message():
        global emergency_message
        message = simpledialog.askstring("Set Emergency Message", "Enter the message for emergencies:")
        if message:
            emergency_message = message
            save_emergency_message(message)
            messagebox.showinfo("Emergency Message Set", "Emergency message set successfully!")

    # Function to change emergency message
    def change_emergency_message():
        global emergency_message
        message = simpledialog.askstring("Change Emergency Message", "Enter the new message for emergencies:")
        if message:
            emergency_message = message
            save_emergency_message(message)
            messagebox.showinfo("Emergency Message Changed", "Emergency message changed successfully!")

    # Function to display added contacts
    def display_contacts():
        contacts_list = "\n".join([f"{name}: {number}" for name, number in contacts.items()])
        messagebox.showinfo("Added Contacts", f"Added Contacts:\n{contacts_list}")

    #name label
    lbl = tkt.Label(main_window, text="Name", width=11, height=1, bg="#00688B", fg="white", font=('times', 22, ' bold '), relief="raised")
    lbl.place(x=650, y=250)

    #name entry
    global name_entry
    name_entry = tkt.Entry(main_window, width=20, fg="white", bg="#00688B", font=('times', 22, ' bold '), relief="raised")
    name_entry.place(x=900, y=250)

    #number label        
    lbl2 = tkt.Label(main_window, text="Number", width=11, bg="#00688B", fg="white", height=1, font=('times', 22, ' bold '), relief="raised")
    lbl2.place(x=650, y=350)

    #number label
    global number_entry
    number_entry = tkt.Entry(main_window, width=20, fg="white", bg="#00688B", font=('times', 22, ' bold '), relief="raised")
    number_entry.place(x=900, y=350)

    # Add contacts button
    add_button = tkt.Button(main_window, text="Add Contact", bg="#00688B", fg="white",width=16, height=2,font=('times', 18, ' bold '), command=add_contact)
    add_button.place(x=800,y=480)

    # delete contact button
    delete_button = tkt.Button(main_window, text="Delete Contact", bg="#00688B", fg="white", width=16, height=2, font=('times', 18, ' bold '), command=delete_contact)
    delete_button.place(x=800,y=580)

    #added contacts button
    display_button = tkt.Button(main_window, text="Added Contacts",bg="white", fg="#00688B",width=16, height=2, activebackground="Red",font=('times', 18, ' bold '),command=display_contacts)
    display_button.place(x=130, y=100)

    #set message button
    set_message_button = tkt.Button(main_window, text="Set Emergency Message",bg="white", fg="#00688B",width=25, height=2, activebackground="Red",font=('times', 18, ' bold '),command=set_emergency_message)
    set_message_button.place(x=70, y=200)

    #change message button
    change_message_button = tkt.Button(main_window, text="Change Emergency Message",bg="white", fg="#00688B",width=25, height=2, activebackground="Red",font=('times', 18, ' bold '),command=change_emergency_message)
    change_message_button.place(x=70, y=300)

    #sos button
   
    sos_image = Image.open(os.path.join(absolute_path,"images/img.jpeg"))
    sos_image = sos_image.resize((200, 200))
    sos_photo = ImageTk.PhotoImage(sos_image)
    sos_button = tkt.Button(main_window, image=sos_photo, bg="white", fg="#00688B", width=200, height=200, activebackground="Red", command=send_sos)
    sos_button.place(x=150, y=420)
    sos_button.image = sos_photo

    main_window.mainloop()


# function to play music
def music():
    # Initialize Pygame
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption('Soothing Meditation Music ')

    # Set up the directory where your music files are located
    musicDir = os.path.join(absolute_path,"music")

    # List all the music files in the directory
    musicFiles = os.listdir(musicDir)

    # Font for displaying text
    font = pygame.font.Font(None, 24)


    def display_text(text, color, x, y):
            text_surface = font.render(text, True, color)
            screen.blit(text_surface, (x, y))

    # Display available music files
    index = None
    music_playing = False
    running = True
    while running:
        screen.fill((255, 255, 255))
        for i, file in enumerate(musicFiles):
            color = (0, 0, 0)
            rect = (50, 50 + i * 30, 300, 30)
            if index == i:
                color = (255, 0, 0)
                pygame.draw.rect(screen, (200, 200, 200), rect)

            display_text(file, color, 50, 50 + i * 30)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    # Check if any music file is clicked
                    for i, file in enumerate(musicFiles):
                        rect = (50, 50 + i * 30, 300, 30)
                        x, y, w, h = rect
                        px, py = event.pos
                        if x < px < x + w and y < py < y + h:
                            index = i
                            selected_musicFile = musicFiles[index]
                            music_path = os.path.join(musicDir, selected_musicFile)
                            pygame.mixer.music.load(music_path)
                            pygame.mixer.music.play(-1)
                            music_playing = True
                            break
            elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if music_playing:
                            pygame.mixer.music.stop()
                            music_playing = False

    # Quit Pygame
    pygame.quit()

# code for buttons on the main window page
Message2 = tkt.Label(
    window, text="Welcome to Wise Wellness, hope you are doing good!",
    bg="white", fg="#4A708B", width=50,
    height=3, font=('times', 20, 'bold'))
Message2.place(x=250, y=300)

sos = tkt.Button(window, text="SOS",
                    fg="white", bg="#A4D3EE", command=sos,
                    width=20, height=3, activebackground="Red",
                    font=('times', 15, ' bold '))
sos.place(x = 250, y= 500)
face = tkt.Button(window, text="Face Recognition", command = face,
                     fg="white", bg="#A4D3EE",
                    width=20, height=3, activebackground="Red",
                    font=('times', 15, ' bold '))
face.place(x = 550, y= 500)
music = tkt.Button(window, text="Music", command = music,
                     fg="white", bg="#A4D3EE",
                    width=20, height=3, activebackground="Red",
                    font=('times', 15, ' bold '))
music.place(x = 850, y= 500)
window.mainloop()
