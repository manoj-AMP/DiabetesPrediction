from django.shortcuts import render, redirect
from django.contrib.auth.models import User, auth
from django.contrib import messages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .models import DiabetesData

def home(request):
    return render(request, "index.html")

def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(username=username, password=password)

        if user is not None:
            auth.login(request, user)
            return render(request, 'data.html')
        else:
            messages.info(request, 'Invalid credentials')
            return render(request, 'login.html')
    else:
        return render(request, 'login.html')

def register(request):
    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        username = request.POST['username']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        email = request.POST['email']

        if password1 == password2:
            if User.objects.filter(username=username).exists():
                messages.info(request, 'Username already exists')
                return render(request, 'register.html')
            elif User.objects.filter(email=email).exists():
                messages.info(request, 'Email already exists')
                return render(request, 'register.html')
            else:
                user = User.objects.create_user(
                    username=username,
                    password=password1,
                    email=email,
                    first_name=first_name,
                    last_name=last_name
                )
                user.save()
                return redirect('login')
        else:
            messages.info(request, 'Passwords do not match')
            return render(request, 'register.html')
    
    return render(request, 'register.html')

def predict(request):
    if request.method == 'POST':
        pregnancies = int(request.POST['pregnancies'])
        glucose = int(request.POST['glucose'])
        bloodpressure = int(request.POST['bloodpressure'])
        skinthickness = int(request.POST['skinthickness'])
        insulin = int(request.POST['insulin'])
        bmi = float(request.POST['bmi'])
        diabetespedigreefunction = float(request.POST['diabetespedigreefunction'])
        age = int(request.POST['age'])

        df = pd.read_csv("static/dataset/diabetes (1).csv")
        df.dropna(inplace=True)
        X_train = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                      'BMI', 'DiabetesPedigreeFunction', 'Age']]
        y_train = df['Outcome']

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        user_input = [[pregnancies, glucose, bloodpressure, skinthickness,
                       insulin, bmi, diabetespedigreefunction, age]]

        prediction = model.predict(user_input)[0]
        probability = model.predict_proba(user_input)[0][1]  # Probability of having diabetes

        DiabetesData.objects.create(
            Pregnancies=pregnancies,
            Glucose=glucose,
            BloodPressure=bloodpressure,
            SkinThickness=skinthickness,
            Insulin=insulin,
            BMI=bmi,
            DiabetesPedigreeFunction=diabetespedigreefunction,
            Age=age
        )

        # Categorize and suggest remedies
        if probability >= 0.85:
            result_text = "⚠️ Patient is affected by diabetes."
            remedy = [
                "Start insulin or oral diabetes medication as prescribed.",
                "Strictly avoid high-sugar and high-carb foods.",
                "Check blood sugar levels multiple times a day.",
                "Consult an endocrinologist regularly."
            ]
        elif 0.65 <= probability < 0.85:
            result_text = "⚠️ Patient is affected by diabetes but at an earlier stage."
            remedy = [
                "Adopt a low glycemic index diet immediately.",
                "Begin a daily exercise routine like walking or swimming.",
                "Use natural aids like fenugreek and cinnamon in meals.",
                "Have regular health checkups every 3-6 months."
            ]
        elif 0.35 <= probability < 0.65:
            result_text = "✅ Patient is not affected by diabetes but at risk in the future."
            remedy = [
                "Avoid excessive sugar intake and processed foods.",
                "Include fiber-rich foods such as oats and leafy greens.",
                "Stay physically active and maintain a healthy weight.",
                "Limit screen time and improve sleep habits."
            ]
        else:
            result_text = "✅ Patient is not affected by diabetes."
            remedy = [
                "Continue balanced diet and regular hydration.",
                "Get at least 7–8 hours of sleep daily.",
                "Stay active — walking, jogging or yoga helps a lot.",
                "Get a diabetes check once a year for early detection."
            ]

        return render(request, 'predict.html', {
            "data": result_text,
            "pregnancies": pregnancies,
            "glucose": glucose,
            "bloodpressure": bloodpressure,
            "skinthickness": skinthickness,
            "insulin": insulin,
            "bmi": bmi,
            "diabetespedigreefunction": diabetespedigreefunction,
            "age": age,
            "remedy": remedy
        })

    return render(request, 'predict.html')
