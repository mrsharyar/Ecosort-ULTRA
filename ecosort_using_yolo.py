import gradio as gr
from ultralytics import YOLO
from PIL import Image
import torch

# --- 1. AI MODEL INITIALIZATION ---
# Using the stock pre-trained YOLO26x model for testing
try:
    model = YOLO("yolo26x.pt")
    ENGINE_STATUS = "🟢 STOCK YOLO26x ACTIVE"
except Exception as e:
    ENGINE_STATUS = f"❌ ERROR LOADING MODEL: {e}"

# --- 2. THE COMPLETE 80-ELEMENT ECO-RULES DICTIONARY ---
ECO_RULES = {
    'person': {"bin": "N/A", "status": "👤 HUMAN", "tip": "Stay awesome! Not for disposal.", "color": "purple"},
    'bicycle': {"bin": "Donation/Scrap", "status": "🔄 REUSE", "tip": "Donate to charity or metal scrap.", "color": "orange"},
    'car': {"bin": "Scrap Yard", "status": "🚗 VEHICLE", "tip": "Specialized automotive recycling required.", "color": "red"},
    'motorcycle': {"bin": "Scrap Yard", "status": "🏍️ VEHICLE", "tip": "Contains oil and battery. Professional disposal.", "color": "red"},
    'airplane': {"bin": "Aviation Scrap", "status": "✈️ SPECIAL", "tip": "Aluminum hull is highly recyclable.", "color": "red"},
    'bus': {"bin": "Scrap Yard", "status": "🚌 VEHICLE", "tip": "Industrial fluid recycling required.", "color": "red"},
    'train': {"bin": "Industrial Scrap", "status": "🚂 SPECIAL", "tip": "Heavy industrial steel recycling.", "color": "red"},
    'truck': {"bin": "Scrap Yard", "status": "🚚 VEHICLE", "tip": "Commercial recycling standards apply.", "color": "red"},
    'boat': {"bin": "Specialized Facility", "status": "🚤 SPECIAL", "tip": "Fiberglass/metal hulls need special handling.", "color": "red"},
    'traffic light': {"bin": "Municipal E-Waste", "status": "🛑 CITY PROPERTY", "tip": "Report damage to local authorities.", "color": "red"},
    'fire hydrant': {"bin": "N/A", "status": "🧯 CITY PROPERTY", "tip": "Safety equipment. Do not disturb.", "color": "red"},
    'stop sign': {"bin": "Metal Scrap", "status": "🛑 CITY PROPERTY", "tip": "Aluminum signs are recyclable.", "color": "red"},
    'parking meter': {"bin": "Municipal E-Waste", "status": "🅿️ CITY PROPERTY", "tip": "Contains electronics and coins.", "color": "red"},
    'bench': {"bin": "Donation/Bulk", "status": "🪑 FURNITURE", "tip": "Wood/Metal can be refurbished.", "color": "orange"},
    'bird': {"bin": "N/A", "status": "🐦 LIVING", "tip": "A part of our ecosystem. Not waste!", "color": "purple"},
    'cat': {"bin": "N/A", "status": "🐾 ANIMAL", "tip": "Please do not recycle your pets.", "color": "purple"},
    'dog': {"bin": "N/A", "status": "🐾 ANIMAL", "tip": "Loyal companion. Not for disposal.", "color": "purple"},
    'horse': {"bin": "N/A", "status": "🐎 LIVING", "tip": "Living creature. Needs care, not bins.", "color": "purple"},
    'sheep': {"bin": "N/A", "status": "🐑 LIVING", "tip": "Living creature. Not waste.", "color": "purple"},
    'cow': {"bin": "N/A", "status": "🐄 LIVING", "tip": "Living creature. Not waste.", "color": "purple"},
    'elephant': {"bin": "N/A", "status": "🐘 LIVING", "tip": "Protected wildlife. Respect nature.", "color": "purple"},
    'bear': {"bin": "N/A", "status": "🐻 LIVING", "tip": "Wild animal. Keep a safe distance.", "color": "purple"},
    'zebra': {"bin": "N/A", "status": "🦓 LIVING", "tip": "Wild animal. Not waste.", "color": "purple"},
    'giraffe': {"bin": "N/A", "status": "🦒 LIVING", "tip": "Wild animal. Not waste.", "color": "purple"},
    'backpack': {"bin": "Donation", "status": "🎒 REUSE", "tip": "If functional, donate to a student.", "color": "orange"},
    'umbrella': {"bin": "General Trash", "status": "🗑️ WASTE", "tip": "Mixed materials are hard to recycle.", "color": "red"},
    'handbag': {"bin": "Donation", "status": "👜 REUSE", "tip": "Leather and fabric can be repurposed.", "color": "orange"},
    'tie': {"bin": "Clothing Bin", "status": "👔 TEXTILE", "tip": "Send to fabric recyclers.", "color": "orange"},
    'suitcase': {"bin": "Donation", "status": "🧳 REUSE", "tip": "Hard-shell cases are mixed plastic/metal.", "color": "orange"},
    'frisbee': {"bin": "Yellow Bin", "status": "♻️ PLASTIC", "tip": "Check for recycling symbol.", "color": "green"},
    'skis': {"bin": "Donation/Scrap", "status": "⛷️ SPORTS", "tip": "Mixed composite materials. Better for reuse.", "color": "orange"},
    'snowboard': {"bin": "Donation/Scrap", "status": "🏂 SPORTS", "tip": "Composite wood/plastic. Reuse if possible.", "color": "orange"},
    'sports ball': {"bin": "General Trash", "status": "🏀 SPORTS", "tip": "Rubber/synthetic leather are rarely recyclable.", "color": "red"},
    'kite': {"bin": "General Trash", "status": "🪁 WASTE", "tip": "Strings can tangle machinery.", "color": "red"},
    'baseball bat': {"bin": "Donation", "status": "⚾ SPORTS", "tip": "Aluminum is highly recyclable.", "color": "orange"},
    'baseball glove': {"bin": "Donation", "status": "⚾ SPORTS", "tip": "Leather items last decades if donated.", "color": "orange"},
    'skateboard': {"bin": "Donation", "status": "🛹 SPORTS", "tip": "Wood/Metal components can be salvaged.", "color": "orange"},
    'surfboard': {"bin": "Specialized Reuse", "status": "🏄 SPORTS", "tip": "Repair resin and foam; difficult to recycle.", "color": "orange"},
    'tennis racket': {"bin": "Donation/Scrap", "status": "🎾 SPORTS", "tip": "Alloy frames are recyclable metal.", "color": "orange"},
    'bottle': {"bin": "Yellow Bin", "status": "♻️ RECYCLE", "tip": "Empty and rinse. Keep the cap on.", "color": "green"},
    'wine glass': {"bin": "Green Bin", "status": "♻️ RECYCLE", "tip": "Rinse out liquids. Handle carefully.", "color": "green"},
    'cup': {"bin": "Yellow/Brown Bin", "status": "⚠️ CHECK", "tip": "Plastic is recycle. Paper often has wax.", "color": "orange"},
    'fork': {"bin": "Blue Bin (Metal)", "status": "♻️ RECYCLE", "tip": "Stainless steel is valuable scrap.", "color": "green"},
    'knife': {"bin": "Blue Bin (Metal)", "status": "♻️ RECYCLE", "tip": "Wrap sharp edges in cardboard.", "color": "green"},
    'spoon': {"bin": "Blue Bin (Metal)", "status": "♻️ RECYCLE", "tip": "Cutlery is infinitely recyclable.", "color": "green"},
    'bowl': {"bin": "Check Material", "status": "🥣 KITCHEN", "tip": "Ceramic is trash. Plastic/Metal is recycle.", "color": "orange"},
    'banana': {"bin": "Brown Bin", "status": "🌱 COMPOST", "tip": "Peels make excellent fertilizer.", "color": "green"},
    'apple': {"bin": "Brown Bin", "status": "🌱 COMPOST", "tip": "Core and seeds are compostable.", "color": "green"},
    'sandwich': {"bin": "Brown Bin", "status": "🌱 COMPOST", "tip": "Food waste only. No plastic wrap.", "color": "green"},
    'orange': {"bin": "Brown Bin", "status": "🌱 COMPOST", "tip": "Citrus peels are great for compost.", "color": "green"},
    'broccoli': {"bin": "Brown Bin", "status": "🌱 COMPOST", "tip": "All vegetable scraps are compostable.", "color": "green"},
    'carrot': {"bin": "Brown Bin", "status": "🌱 COMPOST", "tip": "All vegetable scraps are compostable.", "color": "green"},
    'hot dog': {"bin": "Brown Bin", "status": "🌱 COMPOST", "tip": "Meat waste belongs in organic bins.", "color": "green"},
    'pizza': {"bin": "Brown Bin", "status": "🌱 COMPOST", "tip": "Greasy boxes are compostable!", "color": "green"},
    'donut': {"bin": "Brown Bin", "status": "🌱 COMPOST", "tip": "Bakery waste is organic.", "color": "green"},
    'cake': {"bin": "Brown Bin", "status": "🌱 COMPOST", "tip": "Bakery waste is organic.", "color": "green"},
    'chair': {"bin": "Donation", "status": "🪑 FURNITURE", "tip": "Donate to second-hand stores.", "color": "orange"},
    'couch': {"bin": "Bulk Pickup", "status": "🛋️ FURNITURE", "tip": "Contact city for heavy item pickup.", "color": "orange"},
    'potted plant': {"bin": "Brown Bin", "status": "🪴 ORGANIC", "tip": "Compost the plant; recycle the pot.", "color": "green"},
    'bed': {"bin": "Bulk Pickup", "status": "🛌 FURNITURE", "tip": "Mattresses require special centers.", "color": "orange"},
    'dining table': {"bin": "Donation", "status": "🍽️ FURNITURE", "tip": "Wood/Metal items are great for reuse.", "color": "orange"},
    'toilet': {"bin": "Bulk/Construction", "status": "🚽 SPECIAL", "tip": "Porcelain is used for road base.", "color": "red"},
    'tv': {"bin": "E-Waste Center", "status": "🛑 E-WASTE", "tip": "Contains lead and mercury.", "color": "red"},
    'laptop': {"bin": "E-Waste Center", "status": "🛑 E-WASTE", "tip": "Lithium batteries are fire hazards.", "color": "red"},
    'mouse': {"bin": "E-Waste Center", "status": "🛑 E-WASTE", "tip": "Small electronics have valuable gold.", "color": "red"},
    'remote': {"bin": "E-Waste Center", "status": "🛑 E-WASTE", "tip": "Remove batteries first.", "color": "red"},
    'keyboard': {"bin": "E-Waste Center", "status": "🛑 E-WASTE", "tip": "Circuit boards have precious metals.", "color": "red"},
    'cell phone': {"bin": "E-Waste Center", "status": "🛑 E-WASTE", "tip": "90% of a phone is recyclable.", "color": "red"},
    'microwave': {"bin": "Appliance Center", "status": "🛑 E-WASTE", "tip": "Never put in standard trash.", "color": "red"},
    'oven': {"bin": "Appliance Center", "status": "🛑 E-WASTE", "tip": "Metal body is highly recyclable.", "color": "red"},
    'toaster': {"bin": "Appliance Center", "status": "🛑 E-WASTE", "tip": "Belongs in e-waste bins.", "color": "red"},
    'sink': {"bin": "Construction Scrap", "status": "🚰 SPECIAL", "tip": "Stainless steel is high-value scrap.", "color": "orange"},
    'refrigerator': {"bin": "Appliance Center", "status": "🛑 E-WASTE", "tip": "Must drain coolant (Freon).", "color": "red"},
    'book': {"bin": "Donation/Blue Bin", "status": "📚 REUSE", "tip": "Donate first! Paper is recyclable.", "color": "green"},
    'clock': {"bin": "E-Waste/Trash", "status": "⏰ MISC", "tip": "Electronic = e-waste; Plastic = trash.", "color": "orange"},
    'vase': {"bin": "General Trash", "status": "🏺 CERAMIC", "tip": "Treated glass does not melt like bottles.", "color": "red"},
    'scissors': {"bin": "Blue Bin (Metal)", "status": "✂️ RECYCLE", "tip": "Metal blades are recyclable.", "color": "green"},
    'teddy bear': {"bin": "Donation", "status": "🧸 REUSE", "tip": "Clean and donate to charity.", "color": "orange"},
    'hair drier': {"bin": "E-Waste Center", "status": "🛑 E-WASTE", "tip": "Contains copper wiring.", "color": "red"},
    'toothbrush': {"bin": "Black Bin", "status": "🗑️ TRASH", "tip": "Nylon/Plastics cannot be separated.", "color": "red"}
}

# --- 3. CORE PROCESSING LOGIC ---
def predict_waste(img, conf):
    if img is None: return None, "Please upload an image."

    # Inference using the stock model
    results = model.predict(source=img, conf=conf)

    # Render detection overlay
    annotated = results[0].plot()
    annotated_rgb = Image.fromarray(annotated[..., ::-1])

    # Build the Smart Eco-Report
    report = f"# ♻️ EcoSort Intelligence\n**Engine:** {ENGINE_STATUS}\n\n---\n"
    found = False
    items_tracked = []

    for box in results[0].boxes:
        found = True

        # Safely extract the class index
        cls_id = int(box.cls.item())
        label = model.names[cls_id].lower()
        info = ECO_RULES.get(label, {"bin": "General Trash", "status": "❓ UNKNOWN", "tip": "Consult local guide.", "color": "gray"})

        # Don't repeat the same item type in the text report
        if label not in items_tracked:
            report += f"## {info['status']}: {label.upper()}\n"
            report += f"**📥 BIN:** {info['bin']}\n\n> {info['tip']}\n\n---\n"
            items_tracked.append(label)

    if not found:
        report += "### 🔍 No items detected. Try another angle."

    return annotated_rgb, report

# --- 4. THE GRADIO UI ---
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# ♻️ EcoSort Ultra (Stock Test Mode)")

    with gr.Row():
        with gr.Column():
            input_view = gr.Image(type="pil", label="Visual Input")
            sens_slider = gr.Slider(0.1, 0.9, value=0.25, label="AI Sensitivity")
            scan_btn = gr.Button("🚀 SCAN ENVIRONMENT", variant="primary")

        with gr.Column():
            output_view = gr.Image(type="pil", label="Computer Vision Output")
            report_view = gr.Markdown()

    scan_btn.click(predict_waste, [input_view, sens_slider], [output_view, report_view])

# --- 5. LAUNCH ---
if __name__ == "__main__":
    print("🌐 Opening Public Tunnel...")
    demo.launch(share=True)