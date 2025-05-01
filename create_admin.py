from app import db, User, app
from werkzeug.security import generate_password_hash

with app.app_context():
    password = "1234567890AaA"
    admin = User.query.filter_by(username="admin").first()
    if admin:
        admin.password = generate_password_hash(password)
        print("🔁 Hasło dla 'admin' zostało zaktualizowane.")
    else:
        admin = User(username="admin", password=generate_password_hash(password))
        db.session.add(admin)
        print("✅ Konto 'admin' zostało utworzone.")
    db.session.commit()
