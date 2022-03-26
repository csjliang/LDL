# coding=utf-8
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pickle
import os
import random

choose_number = 0
window = tk.Tk()
window.resizable(width=False, height=False)
window.update()
window.title('Welcome to This User Study')
window.geometry('1024x768')

# welcome image
canvas = tk.Canvas(window, height=768, width=1024, bg='CornflowerBlue')
image_file = tk.PhotoImage(file='welcome.gif')
image = canvas.create_image(512, 200, image=image_file)
canvas.pack(side='top')

# user information
tk.Label(window, text='User name: ', bg='SteelBlue').place(x=350, y=300)
tk.Label(window, text='Password  : ', bg='SteelBlue').place(x=350, y=350)

var_usr_name = tk.StringVar()
var_usr_name.set('')
entry_usr_name = tk.Entry(window, textvariable=var_usr_name).place(x=450, y=300)
var_usr_pwd = tk.StringVar()
entry_usr_pwd = tk.Entry(window, textvariable=var_usr_pwd, show='*').place(x=450, y=350)

try:
    with open('users_info.pickle', 'rb') as usr_file:
        usrs_info = pickle.load(usr_file)
except FileNotFoundError:
    with open('users_info.pickle', 'wb') as usr_file:
        usrs_info = {'1': '1'}
        pickle.dump(usrs_info, usr_file)

print('Loading data, please wait for a while')

finished_files = []
for dir in os.listdir('./'):
    if 'answer' in dir:
        answer_dir = dir
        for file in os.listdir(answer_dir):
            finished_files.append(file.split('.')[0])

image_dir = './all_images'
name_list = []
for name in os.listdir(image_dir):
    if '.png' in name and name.split('.')[0] not in finished_files:
        name_list.append(name)
random.shuffle(name_list)

print(str(len(finished_files)) + ' files finished, ' + str(len(name_list)) + ' files left')

guide_img_dir = './guide_images'
myimages_guide = []
short_side = 480
for i, name in enumerate(sorted(os.listdir(guide_img_dir))):
    img_path = os.path.join(guide_img_dir, name)
    im = Image.open(img_path)
    (x, y) = im.size
    myimages_guide.append(ImageTk.PhotoImage(im.resize((short_side, int(y / (x / short_side))), Image.ANTIALIAS)))

myimagesA = []
# test_number = 10
test_number = len(name_list)
short_side = 950
for i, name in enumerate(name_list[:test_number]):
    img_path = os.path.join(image_dir, name)
    im = Image.open(img_path)
    (x, y) = im.size
    myimagesA.append(ImageTk.PhotoImage(im.resize((short_side, int(y / (x / short_side))), Image.ANTIALIAS)))
    print('data loaded: %10.8s%s' % (str((i + 1) / len(os.listdir(image_dir)) * 100), '%'), end='\r')

print('*'*10, ' loading finished! ', '*'*10)

def guide():

    window_guide = tk.Toplevel(window)
    window_guide.resizable(width=False, height=False)
    window_guide.update()
    window_guide.title('Guidance of the User Study')
    window_guide.geometry('1200x700')
    canvas = tk.Canvas(window_guide, width=1200, height=700, bg='PapayaWhip')
    image = canvas.create_image(60, 80, anchor='nw', image=myimages_guide[0])
    image2 = canvas.create_image(640, 80, anchor='nw', image=myimages_guide[1])
    image3 = canvas.create_image(60, 370, anchor='nw', image=myimages_guide[2])
    image4 = canvas.create_image(640, 370, anchor='nw', image=myimages_guide[3])
    canvas.pack(side='top')
    tk.Label(window_guide,
             text='In each figure, the left one shows neighboring semantics and the right one (with 128*128 pixels) is the patch to be tested.',
             bg='yellow').place(x=200, y=20)

    tk.Label(window_guide,
             text='There are 6 degrees, 0 for no artifacts, and 5 for the most severe cases (see examples). Others can be assigned as 1~4 according to your judgment (larger value = more artifacts).',
             bg='yellow').place(x=20, y=40)

    tk.Label(window_guide, text='The above two look natural, so we may select 0 or 1, indicating that there are no artifacts or negligible artifacts.', bg='Khaki').place(x=260, y=320)
    tk.Label(window_guide, text='This kind of artifacts may be assigned as 5 or 4, indicating severe corruptions (very bad visual looking)', bg='Khaki').place(x=300, y=610)

    btn_login = tk.Button(window_guide, text='Got it, Start!', bg='Khaki', command=select_score_one_image)
    btn_login.place(x=600, y=650)

def select_score_one_image():

    window_test1 = tk.Toplevel(window)
    window_test1.resizable(width=False, height=False)
    window_test1.update()
    window_test1.title('User Study of Artifacts')
    window_test1.geometry('1200x700')
    canvas = tk.Canvas(window_test1, width=1200, height=700, bg='PapayaWhip')
    image = canvas.create_image(120, 40, anchor='nw', image=myimagesA[0])
    canvas.pack(side='top')

    tk.Label(window_test1, text='Left: neighboring semantics (384x384); Right: patches to be tested (128x128)', bg='Khaki').place(x=120, y=10)
    tk.Label(window_test1, text='What degree of artifacts do you think this image (red box) has?', bg='Khaki').place(x=250, y=530)

    var = tk.StringVar()

    l = tk.Label(window_test1, bg='DarkKhaki', width=25, text="You haven't choose answer!!!")
    l.place(x=700, y=530)
    l1 = tk.Label(window_test1, bg='DarkKhaki', width=45,
                  text="Total tests left: " + str(len(myimagesA)))
    l1.place(x=800, y=650)

    def print_selection(event=None):
        if event is not None:
            var.set(str(event.char))
        l.config(text='You have selected ' + var.get())

    def print_test_number():
        global choose_number
        if choose_number == len(myimagesA) - 1:
            l1.config(text='All Finished, Thanks! (click break)')
        if choose_number >= 0 & choose_number < len(myimagesA) - 1:
            # l1.config(text='Already: ' + str(choose_number + 1) + '; Left: ' + str(len(myimagesA)-choose_number-1))
            l1.config(text='Already: ' + str(1398-(len(myimagesA)-choose_number-1)) + '; Left: ' + str(len(myimagesA)-choose_number-1))


    def print_test_number_last():
        global choose_number
        if choose_number == len(myimagesA) - 1:
            l1.config(text='All Finished, Thanks! (click break)')
        if choose_number >= 0 & choose_number < len(myimagesA) - 1:
            # l1.config(text='Already: ' + str(choose_number - 1) + '; Left: ' + str(len(myimagesA)-choose_number+1))
            l1.config(text='Already: ' + str(1398-(len(myimagesA)-choose_number+1)) + '; Left: ' + str(len(myimagesA)-choose_number+1))

    def next_test(event=None):
        print_test_number()
        global choose_number
        os.makedirs('answers_' + var_usr_name.get(), exist_ok=True)
        if var.get() in ['0', '1', '2', '3', '4', '5']:
            os.chdir('answers_' + var_usr_name.get())
            with open(name_list[choose_number].split('.')[0] + '.txt', 'w') as f:
                f.write(name_list[choose_number] + ' Answer: ' + var.get())
            os.chdir(os.path.dirname(os.getcwd()))

            if choose_number <= len(myimagesA) - 2:
                l.config(text='Please select')
                canvas.itemconfig(image, image=myimagesA[choose_number + 1])
                var.set(value=-1)
                choose_number += 1

            if choose_number == len(myimagesA) - 1:
                l1.config(text='All Finished, Thanks! (click break)')
        else:
            l.config(text="Please Choose Your Answer!")

    def last_test(event=None):
        print_test_number_last()
        global choose_number
        if choose_number > 0 and choose_number <= len(myimagesA) - 2:
            l.config(text='Please select')
            canvas.itemconfig(image, image=myimagesA[choose_number - 1])
            var.set(value=-1)
            choose_number -= 1

        if choose_number == 0:
            l1.config(text='This is the first image!')

        if choose_number == len(myimagesA) - 1:
            l1.config(text='All Finished, Thanks!')

    r1 = tk.Radiobutton(window_test1, text='0 (No Artifacts)',
                        variable=var, value='0',
                        command=print_selection)
    r1.place(x=440, y=575)
    r1.bind_all("0", print_selection)
    r2 = tk.Radiobutton(window_test1, text='2',
                        variable=var, value='2',
                        command=print_selection)
    r2.place(x=440, y=610)
    r2.bind_all("2", print_selection)
    r3 = tk.Radiobutton(window_test1, text='4',
                        variable=var, value='4',
                        command=print_selection)
    r3.place(x=440, y=645)
    r3.bind_all("4", print_selection)
    r4 = tk.Radiobutton(window_test1, text='1',
                        variable=var, value='1',
                        command=print_selection)
    r4.place(x=580, y=575)
    r4.bind_all("1", print_selection)
    r5 = tk.Radiobutton(window_test1, text='3',
                        variable=var, value='3',
                        command=print_selection)
    r5.place(x=580, y=610)
    r5.bind_all("3", print_selection)
    r6 = tk.Radiobutton(window_test1, text='5 (Worst!!)',
                        variable=var, value='5',
                        command=print_selection)
    r6.place(x=580, y=645)
    r6.bind_all("5", print_selection)
    btn_login = tk.Button(window_test1, text='previous', bg='Khaki', command=last_test)
    btn_login.place(x=900, y=610)
    btn_login.bind_all('<Left>', last_test)
    btn_login = tk.Button(window_test1, text='next', bg='Khaki', command=next_test)
    btn_login.place(x=700, y=610)
    btn_login.bind_all('<Right>', next_test)
    btn_login = tk.Button(window_test1, text='break', bg='Khaki', command=window.destroy)
    btn_login.place(x=1000, y=610)

def usr_login():
    usr_name = var_usr_name.get()
    usr_pwd = var_usr_pwd.get()
    try:
        with open('users_info.pickle', 'rb') as usr_file:
            usrs_info = pickle.load(usr_file)
    except FileNotFoundError:
        with open('usres_info.pickle', 'wb') as usr_file:
            usrs_info = {'1': '1'}
            pickle.dump(usrs_info, usr_file)
    if usr_name in usrs_info:
        if usr_pwd == usrs_info[usr_name]:
            global choose_number
            choose_number = 0
            # select_score_one_image()
            guide()
        else:
            tk.messagebox.showerror(message='Error, your password is wrong, try again.')
    else:
        is_sign_up = tk.messagebox.askyesno('Welcome', 'Please sign up or use public name and password: {1:1}')
        if is_sign_up:
            usr_sign_up()


def usr_sign_up():
    def sign_to_Mofan_Python():
        np = new_pwd.get()
        npf = new_pwd_confirm.get()
        nn = new_name.get()
        with open('users_info.pickle', 'rb') as usr_file:
            exist_usr_info = pickle.load(usr_file)
        if np != npf:
            tk.messagebox.showerror('Error', 'Password and confirm password must be the same!')
        elif nn in exist_usr_info:
            tk.messagebox.showerror('Error', 'The user has already signed up!')
        else:
            exist_usr_info[nn] = np
            with open('users_info.pickle', 'wb') as usr_file:
                pickle.dump(exist_usr_info, usr_file)
            tk.messagebox.showinfo('Welcome', 'You have successfully signed up!')
            window_sign_up.destroy()

    window_sign_up = tk.Toplevel(window)
    window_sign_up.geometry('350x200')
    window_sign_up.title('Sign up window')

    new_name = tk.StringVar()
    new_name.set('')
    tk.Label(window_sign_up, text='User name: ').place(x=10, y=10)
    entry_new_name = tk.Entry(window_sign_up, textvariable=new_name, fg='LightGrey')
    entry_new_name.place(x=150, y=10)

    new_pwd = tk.StringVar()
    tk.Label(window_sign_up, text='Password: ').place(x=10, y=50)
    entry_usr_pwd = tk.Entry(window_sign_up, textvariable=new_pwd, show='*')
    entry_usr_pwd.place(x=150, y=50)

    new_pwd_confirm = tk.StringVar()
    tk.Label(window_sign_up, text='Confirm password: ').place(x=10, y=90)
    entry_usr_pwd_confirm = tk.Entry(window_sign_up, textvariable=new_pwd_confirm, show='*')
    entry_usr_pwd_confirm.place(x=150, y=90)

    btn_comfirm_sign_up = tk.Button(window_sign_up, text='Sign up', command=sign_to_Mofan_Python)
    btn_comfirm_sign_up.place(x=150, y=130)

# login and sign up button
btn_login = tk.Button(window, text='Login', bg='SteelBlue', command=usr_login)
btn_login.place(x=400, y=420)
btn_sign_up = tk.Button(window, text='Sign up', bg='SteelBlue', command=usr_sign_up)
btn_sign_up.place(x=500, y=420)

window.mainloop()






