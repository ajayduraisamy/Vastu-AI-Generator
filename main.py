
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, uuid, os, torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from PIL import Image, ImageDraw, ImageFont





app = Flask(__name__)
CORS(app)

# --- CONFIG ---
DB_NAME = "database.db"
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "generated"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


DEVICE = "cpu" 

# --- DATABASE SETUP ---
def init_db():
    db = sqlite3.connect(DB_NAME)
    # Full table with name, email, number, password, address
    db.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        name TEXT, email TEXT UNIQUE, number TEXT, password TEXT, address TEXT)""")
    db.execute("""CREATE TABLE IF NOT EXISTS plans(
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, 
        width TEXT, height TEXT, image_path TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    db.commit()
    db.close()

init_db()
# --- SDXL-TURBO MODELS ---
print(f"Loading SDXL-Turbo on {DEVICE}...")
model_id = "stabilityai/sdxl-turbo"
t_type = torch.float16 if DEVICE == "cuda" else torch.float32

# Load Text-to-Image and Image-to-Image for home photo processing
txt2img = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=t_type, variant="fp16" if DEVICE == "cuda" else None).to(DEVICE)
img2img = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=t_type, variant="fp16" if DEVICE == "cuda" else None).to(DEVICE)

# --- PREMIUM LABEL ENGINE ---
def add_premium_labels(img_path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arialbd.ttf", 22) # Ensure this font exists in your dir
    except:
        font = ImageFont.load_default()

    # Vastu Positions
    coords = {
        "POOJA (NE)": (370, 50),
        "KITCHEN (SE)": (370, 410),
        "BEDROOM (SW)": (50, 410),
        "TOILET (NW)": (50, 50),
        "LIVING (CTR)": (210, 230)
    }

    for label, pos in coords.items():
        text_w = draw.textlength(label, font=font)
        draw.rectangle([pos[0]-5, pos[1]-5, pos[0]+text_w+5, pos[1]+25], fill="white", outline="black", width=2)
        draw.text(pos, label, fill="black", font=font)
    img.save(img_path)

# --- API FOR FLUTTER ---

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    print("Registering user:", data)
    db = sqlite3.connect(DB_NAME)
    check = db.execute("SELECT * FROM users WHERE email=?", (data['email'],)).fetchone()
    if check:
        db.close()
        return jsonify({"status": "error", "message": "Email already exists"}), 400
    try:
        db.execute("INSERT INTO users (name, email, number, password, address) VALUES (?,?,?,?,?)", 
                   (data['name'], data['email'], data['number'], generate_password_hash(data['password']), data['address']))
        db.commit()
        return jsonify({"status": "success"}), 201
    except: return jsonify({"status": "error"}), 400
    finally: db.close()

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    db = sqlite3.connect(DB_NAME)
    db.row_factory = sqlite3.Row
    user = db.execute("SELECT * FROM users WHERE email=?", (data['email'],)).fetchone()
    db.close()
    if user and check_password_hash(user["password"], data["password"]):
        return jsonify({"status": "success", "user": dict(user)})
    return jsonify({"status": "error"}), 401

@app.route("/profile/<int:user_id>", methods=["GET"])
def profile(user_id):
    db = sqlite3.connect(DB_NAME)
    db.row_factory = sqlite3.Row
    user = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    db.close()
    if user:
        return jsonify({"status": "success", "user": dict(user)})
    return jsonify({"status": "error"}), 404

@app.route("/generate", methods=["POST"])
def generate():
    user_id = request.form.get("user_id")
    w, h = request.form.get("width", "30"), request.form.get("height", "40")
    
    # Smarter prompt to tell AI to fix the UPLOADED layout
    prompt = (f"Masterpiece architectural CAD revision, transform this sketch into a "
              f"professional 2D blueprint, {w}x{h} plot, top-down view, sharp black lines, "
              f"clean white background, high contrast technical drawing, maintain original wall structure")
    
    filename = f"{uuid.uuid4()}.png"
    out_path = os.path.join(OUTPUT_DIR, filename)

    if 'home_photo' in request.files and request.files['home_photo'].filename != '':
        file = request.files['home_photo']
        up_path = os.path.join(UPLOAD_DIR, f"ref_{filename}")
        file.save(up_path)
        
        init_image = Image.open(up_path).convert("RGB").resize((512, 512))
        
        # LOWER STRENGTH (0.35) keeps YOUR walls but fixes the DRAWING quality
        image = img2img(
            prompt=prompt, 
            image=init_image, 
            strength=0.35, 
            num_inference_steps=5, 
            guidance_scale=8.5 # Higher guidance for sharper lines
        ).images[0]
    else:
        image = txt2img(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]

    image.save(out_path)
    
    # Now apply the labels
    add_premium_labels(out_path)

    
    db = sqlite3.connect(DB_NAME)
    db.execute("INSERT INTO plans (user_id, width, height, image_path) VALUES (?,?,?,?)", (user_id, w, h, filename))
    db.commit()
    db.close()
    
    return jsonify({"image_url": f"/output/{filename}"}), 200

@app.route("/history/<int:user_id>", methods=["GET"])
def history(user_id):
    db = sqlite3.connect(DB_NAME)
    db.row_factory = sqlite3.Row
    rows = db.execute("SELECT * FROM plans WHERE user_id=?", (user_id,)).fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route("/output/<filename>")
def serve_image(filename):
    return send_file(os.path.join(OUTPUT_DIR, filename))

if __name__ == "__main__":
    # use_reloader=False is mandatory to prevent MemoryError during model load
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)