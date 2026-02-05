from flask import Flask, request, send_file, render_template, redirect, session, url_for
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, uuid, os, torch
from diffusers import AutoPipelineForText2Image, EulerAncestralDiscreteScheduler
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
CORS(app)
app.secret_key = "vastu_premium_key"

# --- CONFIG ---
DB_NAME = "database.db"
OUTPUT_DIR = "generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use CUDA for lightning speed (< 5 seconds)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- DATABASE ---
def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
    db.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT UNIQUE, password TEXT)")
    db.commit()
    db.close()

init_db()

# --- TURBO AI MODEL (FASTER & CLEARER) ---
print(f"Loading SDXL-Turbo on {DEVICE}...")
# SDXL-Turbo is much sharper for CAD lines than SD 1.5
model_id = "stabilityai/sdxl-turbo" 
pipe = AutoPipelineForText2Image.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    variant="fp16" if DEVICE == "cuda" else None
).to(DEVICE)

# --- PREMIUM LABEL ENGINE ---
def add_premium_labels(img_path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arialbd.ttf", 20) # Bold for clarity
    except:
        font = ImageFont.load_default()

    # Vastu Positions (Calibrated for SDXL 512x512)
    coords = {
        "POOJA (NE)": (370, 50),
        "KITCHEN (SE)": (370, 410),
        "BEDROOM (SW)": (50, 410),
        "TOILET (NW)": (50, 50),
        "LIVING (CTR)": (210, 230)
    }

    for label, pos in coords.items():
        # High-contrast premium label design
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
    return 'Login Page: <form method="post"><input name="email"><input type="password" name="password"><button>Login</button></form>'

@app.route("/generate")
def generate_ui():
    if "uid" not in session: return redirect(url_for('login'))
    return render_template("generate.html", name=session["name"])

@app.route("/generate-plan", methods=["POST"])
def generate_plan():
    if "uid" not in session: return redirect(url_for('login'))
    
    w, h = request.form["width"], request.form["height"]
    # Sharp CAD-specific prompt
    prompt = f"Architectural 2D CAD floor plan, black and white, professional blueprint, {w}x{h} plot, top-down view, highly detailed sharp lines, technical drawing, no furniture, white background"
    
    filename = f"{uuid.uuid4()}.png"
    out_path = os.path.join(OUTPUT_DIR, filename)

    # Turbo only needs 1-4 steps to be crystal clear
    image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
    image.save(out_path)

    add_premium_labels(out_path)
    return render_template("generate.html", name=session["name"], result_img=filename)

@app.route("/output/<img>")
def send_output(img):
    return send_file(os.path.join(OUTPUT_DIR, img))

if __name__ == "__main__":
    app.run(debug=True, port=5000)