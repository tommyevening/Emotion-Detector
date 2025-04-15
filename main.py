import tkinter as tk
from tkinter import messagebox

def open_new_window():
    new_window = tk.Toplevel(app)
    new_window.title("New Window")
    new_window.geometry("250x150")
    new_window.config(bg="#26242f")
    cancel_button = tk.Button(new_window, text="Cancel", command=new_window.destroy)
    cancel_button.pack(pady=10)
    ok_button = tk.Button(new_window, text="OK", command=new_window.destroy)
    ok_button.pack(pady=10)


    tk.Label(new_window, text="This is a new window").pack(pady=20)

app = tk.Tk()

app.title("Mood Scanner")
app.geometry("1024x720+150+50")
app.config(bg="#26242f")


drop_photo = tk.PhotoImage(file="assets/test/angry/PrivateTest_88305.jpg")
apply_button = tk.Button(app, text="Apply", font=("Arial", 20), bg="#3a3a3a", fg="white", width=10)
apply_button.pack(pady=20)
apply_button.config(command=open_new_window)




app.mainloop()