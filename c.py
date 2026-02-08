from flask import Flask, request, send_file, render_template, redirect, session, url_for
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, uuid, os, torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
CORS(app)
app.secret_key = "vastu_premium_key"

# --- CONFIG ---
DB_NAME = "database.db"
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "generated"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- DATABASE ---
def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
    db.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT UNIQUE, password TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS plans(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, plot_width TEXT, plot_height TEXT, image_path TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
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

# --- ROUTES ---
@app.route("/")
def home(): return redirect(url_for('login'))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE email=?", (request.form["email"],)).fetchone()
        if user and check_password_hash(user["password"], request.form["password"]):
            session["uid"], session["name"] = user["id"], user["name"]
            return redirect(url_for('generate_ui'))
    return 'Login: <form method="post"><input name="email"><input type="password" name="password"><button>Login</button></form>'

@app.route("/generate")
def generate_ui():
    if "uid" not in session: return redirect(url_for('login'))
    return render_template("generate.html", name=session["name"])

# --- ENHANCED GENERATION ENGINE ---
@app.route("/generate-plan", methods=["POST"])
def generate_plan():
    if "uid" not in session: return redirect(url_for('login'))
    
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
    
    return render_template("generate.html", name=session["name"], result_img=filename)

@app.route("/output/<img>")
def send_output(img):
    return send_file(os.path.join(OUTPUT_DIR, img))

if __name__ == "__main__":
    # use_reloader=False prevents the double-loading/breaking issue
    app.run(debug=True, port=5000, use_reloader=False)
