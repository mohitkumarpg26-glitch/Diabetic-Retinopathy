import kagglehub

kagglehub.login(validate_credentials=False)

try:
    path = kagglehub.competition_download("aptos2019-blindness-detection")
    print("Downloaded to:", path)
except Exception as error:
    response = getattr(error, "response", None)
    print("HTTP status:", getattr(response, "status_code", None))
    details = getattr(response, "text", "") or str(error)
    print("Kaggle details:", details[:1000])
