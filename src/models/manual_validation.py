import joblib
import pandas as pd
import os

def run_manual_validation():
    print("Loading model for manual validation...")
    model_path = "models/final_pipeline.joblib"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run serialize.py first.")
        return
        
    model = joblib.load(model_path)
    
    examples = [
        {
            "id": "A",
            "type": "Legitimate email with URLs",
            "Expected": 0,
            "Subject": "GitHub Password Reset",
            "Body": "Reset your GitHub password using https://github.com/password_reset?token=12345. If you did not request this, please ignore this email."
        },
        {
            "id": "B",
            "type": "Legitimate newsletter containing multiple URLs",
            "Expected": 0,
            "Subject": "Weekly Python Newsletter",
            "Body": "Welcome to this week's Python news! Read our latest article at http://python.org/news. Also check out the new library at http://github.com/python/lib. Thanks for subscribing!"
        },
        {
            "id": "C",
            "type": "Legitimate organization announcement with urgency",
            "Expected": 0,
            "Subject": "ACTION REQUIRED: Update your employee benefits",
            "Body": "Action required by all employees. Please verify your account information and update your benefits immediately before the deadline on Friday."
        },
        {
            "id": "D",
            "type": "Phishing email without any URL",
            "Expected": 1,
            "Subject": "Your account has been suspended",
            "Body": "We have suspended your account due to suspicious activity. Please reply to this email immediately with your password to verify your identity. Urgent action required."
        },
        {
            "id": "E",
            "type": "Phishing email using a bare domain",
            "Expected": 1,
            "Subject": "Security Alert: Login from new device",
            "Body": "A new device logged into your account. If this was not you, please secure your account by visiting login-secure-update.com/verify."
        },
        {
            "id": "F",
            "type": "Phishing email with no obvious urgency",
            "Expected": 1,
            "Subject": "Invoice #49102 attached",
            "Body": "Please find the attached invoice for last month's services. You can download the PDF here: [URL]"
        },
        {
            "id": "G",
            "type": "Normal personal email",
            "Expected": 0,
            "Subject": "Dinner tonight?",
            "Body": "Hey, are we still on for dinner tonight at 7? Let me know if you need to reschedule. See you!"
        },
        {
            "id": "H",
            "type": "HTML-style legitimate email",
            "Expected": 0,
            "Subject": "Your recent order receipt",
            "Body": "<html><body><h1>Order Confirmation</h1><p>Thank you for your order! Track your package <a href='http://store.com/track'>here</a>.</p></body></html>"
        }
    ]
    
    df = pd.DataFrame(examples)
    
    print("Running predictions on external manual validation set...")
    predictions = model.predict(df[['Subject', 'Body']])
    probabilities = model.predict_proba(df[['Subject', 'Body']])
    
    results = []
    for i, row in df.iterrows():
        pred = predictions[i]
        prob = probabilities[i][1] if pred == 1 else probabilities[i][0]
        
        results.append({
            "ID": row["id"],
            "Type": row["type"],
            "Expected_Label": row["Expected"],
            "Predicted_Label": pred,
            "Probability": f"{prob*100:.2f}%",
            "Match": row["Expected"] == pred
        })
        
    res_df = pd.DataFrame(results)
    
    os.makedirs("results", exist_ok=True)
    res_df.to_csv("results/manual_validation.csv", index=False)
    
    print("Manual validation complete. Results saved to results/manual_validation.csv")
    print(res_df.to_string())

if __name__ == "__main__":
    run_manual_validation()
