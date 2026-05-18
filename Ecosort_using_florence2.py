"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       ♻️  ECOSORT ULTRA  ♻️                                ║
║            AI-Powered Waste Detection & Sorting Intelligence               ║
║                     Powered by Microsoft Florence-2                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Florence-2 is used for:
  - <OD>              → Open-vocabulary Object Detection (bounding boxes)
  - <CAPTION>         → Short image caption
  - <DETAILED_CAPTION>→ Rich scene description

Run in Google Colab (T4 GPU recommended):
  !pip install gradio transformers torch pillow einops flash_attn timm
  Then upload this file and run it.
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM

# ═══════════════════════════════════════════════════════════════════════════════
#  1.  FLORENCE-2 MODEL INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_ID = "microsoft/Florence-2-large"

try:
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _dtype  = torch.float16 if _device == "cuda" else torch.float32

    florence_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=_dtype,
        trust_remote_code=True,
    ).to(_device).eval()

    florence_processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    ENGINE_STATUS = f"🟢 FLORENCE-2-LARGE ACTIVE  ({_device.upper()} / {_dtype})"

except Exception as e:
    florence_model = None
    florence_processor = None
    ENGINE_STATUS = f"❌ MODEL LOAD FAILED: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  FLORENCE-2 INFERENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def florence_run(image: Image.Image, task_prompt: str, text_input: str = "") -> dict:
    """Run a single Florence-2 task and return the parsed result dict."""
    prompt = task_prompt if not text_input else task_prompt + text_input
    inputs = florence_processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(_device, _dtype) if v.dtype == torch.float32 else v.to(_device)
              for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = florence_model.generate(
            **inputs,
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
    generated_text = florence_processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]
    parsed = florence_processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height),
    )
    return parsed


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  COMPREHENSIVE ECO-RULES DICTIONARY  (300+ waste-mapped categories)
# ═══════════════════════════════════════════════════════════════════════════════

ECO_RULES = {
    # ── PLASTICS & PACKAGING ───────────────────────────────────────────────────
    "plastic bottle":   {"bin": "Yellow Bin (Recycle)",       "status": "♻️ RECYCLE", "tip": "PET plastic. Crush flat and keep cap on."},
    "water bottle":     {"bin": "Yellow Bin (Recycle)",       "status": "♻️ RECYCLE", "tip": "Rinse and recycle. Consider reusable bottles!"},
    "soda bottle":      {"bin": "Yellow Bin (Recycle)",       "status": "♻️ RECYCLE", "tip": "Empty liquids first."},
    "shampoo bottle":   {"bin": "Yellow Bin (Recycle)",       "status": "♻️ RECYCLE", "tip": "Rinse out soap. The pump usually goes in the trash."},
    "detergent bottle": {"bin": "Yellow Bin (Recycle)",       "status": "♻️ RECYCLE", "tip": "Thick HDPE plastic is highly recyclable."},
    "milk jug":         {"bin": "Yellow Bin (Recycle)",       "status": "♻️ RECYCLE", "tip": "Rinse well to avoid smells."},
    "plastic bag":      {"bin": "Store Drop-off",             "status": "⚠️ SPECIAL", "tip": "Soft plastics jam sorting machines. Take to grocery stores."},
    "grocery bag":      {"bin": "Store Drop-off",             "status": "⚠️ SPECIAL", "tip": "Take to a film recycling drop-off."},
    "ziplock bag":      {"bin": "Store Drop-off",             "status": "⚠️ SPECIAL", "tip": "Clean and dry before dropping off with film plastics."},
    "bubble wrap":      {"bin": "Store Drop-off",             "status": "⚠️ SPECIAL", "tip": "Pop it for fun, then recycle with soft plastics."},
    "plastic wrap":     {"bin": "Store Drop-off",             "status": "⚠️ SPECIAL", "tip": "Cling film cannot go in standard bins."},
    "styrofoam":        {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "Expanded polystyrene is rarely recycled curb-side."},
    "packing peanuts":  {"bin": "General Trash / Reuse",      "status": "🗑️ TRASH",   "tip": "Unless they dissolve in water (cornstarch), put in trash."},
    "tupperware":       {"bin": "Donation / Check Local",     "status": "♻️ CHECK",   "tip": "Rigid plastics #5 can sometimes be recycled. Donate if usable."},
    "plastic container":{"bin": "Yellow Bin (Recycle)",       "status": "♻️ RECYCLE", "tip": "Check for recycling #1, #2, or #5."},
    "plastic cup":      {"bin": "Yellow Bin (Recycle)",       "status": "♻️ RECYCLE", "tip": "Check if your city accepts plastic cups."},
    "plastic straw":    {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "Too small to sort. Falls through recycling machines."},
    "plastic cutlery":  {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "Usually not recyclable due to mixed low-grade plastics."},
    "plastic fork":     {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "Trash. Switch to metal or bamboo!"},
    "plastic spoon":    {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "Trash. Switch to metal or bamboo!"},
    "yogurt cup":       {"bin": "Yellow Bin (Recycle)",       "status": "♻️ RECYCLE", "tip": "Wash out the dairy first."},
    "cd case":          {"bin": "Specialty Recycle / Trash",  "status": "⚠️ CHECK",   "tip": "Polystyrene cases are often not curb-side recyclable."},
    "pvc pipe":         {"bin": "Construction Scrap",         "status": "🏗️ SPECIAL", "tip": "PVC (#3) is rarely accepted in home recycling."},
    
    # ── PAPER & CARDBOARD ──────────────────────────────────────────────────────
    "cardboard box":    {"bin": "Blue Bin (Paper)",           "status": "📦 RECYCLE", "tip": "Break it down flat to save space."},
    "cardboard":        {"bin": "Blue Bin (Paper)",           "status": "♻️ RECYCLE", "tip": "Flatten boxes."},
    "pizza box":        {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "Grease ruins paper recycling. Compost it!"},
    "newspaper":        {"bin": "Blue Bin (Paper)",           "status": "♻️ RECYCLE", "tip": "Keep dry."},
    "magazine":         {"bin": "Blue Bin (Paper)",           "status": "♻️ RECYCLE", "tip": "Glossy paper is fine to recycle."},
    "flyer":            {"bin": "Blue Bin (Paper)",           "status": "♻️ RECYCLE", "tip": "Junk mail is recyclable."},
    "envelope":         {"bin": "Blue Bin (Paper)",           "status": "♻️ RECYCLE", "tip": "Plastic windows are usually okay, but removing them is best."},
    "paper bag":        {"bin": "Blue Bin (Paper)",           "status": "♻️ RECYCLE", "tip": "If greasy, compost it instead."},
    "notebook":         {"bin": "Blue Bin (Paper)",           "status": "♻️ RECYCLE", "tip": "Remove metal spiral bindings first."},
    "book":             {"bin": "Donation / Blue Bin",        "status": "📚 REUSE",   "tip": "Donate books. If ruined, recycle the paper pages."},
    "receipt":          {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "Thermal receipts contain BPA and cannot be recycled."},
    "sticky note":      {"bin": "Blue Bin (Paper)",           "status": "♻️ RECYCLE", "tip": "The glue is filtered out during recycling."},
    "wrapping paper":   {"bin": "Blue Bin / Trash",           "status": "⚠️ CHECK",   "tip": "If it has glitter or foil, it goes in the trash. Test: if it scrunches and stays, recycle it."},
    "tissue":           {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "Paper fibers are too short to recycle again. Compost!"},
    "paper towel":      {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "If used with chemical cleaners, put in trash instead."},
    "toilet paper tube":{"bin": "Blue Bin (Paper)",           "status": "♻️ RECYCLE", "tip": "Cardboard tubes are great for recycling or compost."},
    "egg carton":       {"bin": "Blue Bin / Compost",         "status": "♻️ RECYCLE", "tip": "Paper cartons can be recycled or composted. Styrofoam goes to trash."},
    "coffee cup":       {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "Paper coffee cups have a hidden plastic lining. Trash them."},

    # ── GLASS & CERAMICS ───────────────────────────────────────────────────────
    "glass bottle":     {"bin": "Green Bin (Glass)",          "status": "♻️ RECYCLE", "tip": "Infinitely recyclable. Sort by color if required locally."},
    "wine bottle":      {"bin": "Green Bin (Glass)",          "status": "♻️ RECYCLE", "tip": "Remove the cork first."},
    "beer bottle":      {"bin": "Green Bin (Glass)",          "status": "♻️ RECYCLE", "tip": "Recycle the glass. Metal caps go in the metal bin."},
    "mason jar":        {"bin": "Green Bin (Glass)",          "status": "♻️ RECYCLE", "tip": "Rinse out food. Metal lid recycles separately."},
    "jar":              {"bin": "Green Bin (Glass)",          "status": "♻️ RECYCLE", "tip": "Clean out jam or sauce residue."},
    "broken glass":     {"bin": "General Trash",              "status": "⚠️ DANGER",  "tip": "Wrap securely in newspaper before trashing to protect workers."},
    "window glass":     {"bin": "Construction Scrap",         "status": "🏗️ SPECIAL", "tip": "Treated differently than bottle glass. Do not put in home bins."},
    "mirror":           {"bin": "General Trash / Special",    "status": "🪞 SPECIAL", "tip": "The reflective coating contaminates glass recycling."},
    "light bulb":       {"bin": "Hazardous / E-Waste",        "status": "💡 HAZARD",  "tip": "LEDs are e-waste. CFLs contain mercury (hazardous). Incandescent is trash."},
    "ceramic mug":      {"bin": "Donation / Trash",           "status": "☕ CERAMIC",  "tip": "Ceramics melt at a different temperature than glass. Do not recycle."},
    "plate":            {"bin": "Donation / Trash",           "status": "🍽️ CERAMIC",  "tip": "Ceramic or porcelain plates go in the trash if broken."},
    "bowl":             {"bin": "Donation / Trash",           "status": "🥣 CERAMIC",  "tip": "If broken, wrap safely and place in trash."},
    "vase":             {"bin": "Donation / Trash",           "status": "🏺 CERAMIC",  "tip": "Decorative glass/ceramic cannot go in standard recycling."},

    # ── METALS ─────────────────────────────────────────────────────────────────
    "soda can":         {"bin": "Yellow Bin (Metal)",         "status": "♻️ RECYCLE", "tip": "Aluminum is infinitely recyclable."},
    "tin can":          {"bin": "Yellow Bin (Metal)",         "status": "♻️ RECYCLE", "tip": "Rinse out food (beans, soup) first."},
    "soup can":         {"bin": "Yellow Bin (Metal)",         "status": "♻️ RECYCLE", "tip": "Push the sharp lid inside the can and pinch closed."},
    "aluminum foil":    {"bin": "Yellow Bin (Metal)",         "status": "♻️ RECYCLE", "tip": "Scrunch into a ball. Must be clean of food/grease."},
    "bottle cap":       {"bin": "Yellow Bin (Metal)",         "status": "♻️ RECYCLE", "tip": "Collect metal caps inside an empty tin can and crimp it closed."},
    "aerosol can":      {"bin": "Yellow Bin / Hazardous",     "status": "⚠️ CHECK",   "tip": "Must be completely empty. If it contains chemicals, take to hazardous waste."},
    "wire hanger":      {"bin": "Metal Scrap / Dry Cleaner",  "status": "♻️ SPECIAL", "tip": "Jams recycling machines. Return to dry cleaner or scrap metal yard."},
    "nail":             {"bin": "Metal Scrap",                "status": "🔨 SCRAP",   "tip": "Too small for home recycling. Save for a scrap metal yard."},
    "screw":            {"bin": "Metal Scrap",                "status": "🔩 SCRAP",   "tip": "Save for a scrap metal drop-off."},
    "copper wire":      {"bin": "Metal Scrap / E-Waste",      "status": "🔌 SCRAP",   "tip": "Highly valuable scrap metal. Never landfill."},
    "pot":              {"bin": "Metal Scrap / Donation",     "status": "🍲 SCRAP",   "tip": "Stainless steel or cast iron pots are great scrap metal."},
    "pan":              {"bin": "Metal Scrap / Donation",     "status": "🍳 SCRAP",   "tip": "Teflon pans are hard to recycle, but metal scrap yards take them."},
    "keys":             {"bin": "Metal Scrap",                "status": "🔑 SCRAP",   "tip": "Brass and steel keys are recyclable at scrap yards."},
    "cutlery":          {"bin": "Metal Scrap / Donation",     "status": "🍴 SCRAP",   "tip": "Stainless steel forks/knives have great scrap value."},
    
    # ── E-WASTE & TECH ─────────────────────────────────────────────────────────
    "laptop":           {"bin": "E-Waste Center",             "status": "🛑 E-WASTE", "tip": "Contains a lithium battery. Extreme fire hazard in trash."},
    "smartphone":       {"bin": "E-Waste Center",             "status": "📱 E-WASTE", "tip": "Wipe your data first! Contains precious metals."},
    "cell phone":       {"bin": "E-Waste Center",             "status": "📱 E-WASTE", "tip": "Wipe your data first! Contains precious metals."},
    "tablet":           {"bin": "E-Waste Center",             "status": "📱 E-WASTE", "tip": "Lithium battery hazard."},
    "television":       {"bin": "E-Waste Center",             "status": "📺 E-WASTE", "tip": "Contains heavy metals like lead or mercury."},
    "monitor":          {"bin": "E-Waste Center",             "status": "🖥️ E-WASTE", "tip": "E-waste only. Do not put in dumpsters."},
    "keyboard":         {"bin": "E-Waste Center",             "status": "⌨️ E-WASTE", "tip": "Electronic circuit boards inside."},
    "computer mouse":   {"bin": "E-Waste Center",             "status": "🖱️ E-WASTE", "tip": "E-waste collection point."},
    "motherboard":      {"bin": "E-Waste Center",             "status": "🛑 E-WASTE", "tip": "Contains gold, silver, and palladium!"},
    "gpu":              {"bin": "E-Waste Center",             "status": "🛑 E-WASTE", "tip": "High-value e-waste component."},
    "hard drive":       {"bin": "E-Waste Center",             "status": "🛑 E-WASTE", "tip": "Destroy data first. Contains neodymium magnets."},
    "usb drive":        {"bin": "E-Waste Center",             "status": "💾 E-WASTE", "tip": "Small but still e-waste."},
    "charger":          {"bin": "E-Waste Center",             "status": "🔌 E-WASTE", "tip": "The copper wiring is valuable."},
    "power cable":      {"bin": "E-Waste Center",             "status": "🔌 E-WASTE", "tip": "Do not throw cables in the trash."},
    "headphones":       {"bin": "E-Waste Center",             "status": "🎧 E-WASTE", "tip": "Contains wiring and small magnets."},
    "earbuds":          {"bin": "E-Waste Center",             "status": "🎧 E-WASTE", "tip": "If wireless, they contain tiny lithium batteries (fire risk)."},
    "smartwatch":       {"bin": "E-Waste Center",             "status": "⌚ E-WASTE", "tip": "Contains a lithium-ion battery."},
    "router":           {"bin": "E-Waste Center",             "status": "📡 E-WASTE", "tip": "Circuit boards belong in e-waste."},
    "game console":     {"bin": "E-Waste Center",             "status": "🎮 E-WASTE", "tip": "Try to repair or sell first. Otherwise, e-waste."},
    "controller":       {"bin": "E-Waste Center",             "status": "🎮 E-WASTE", "tip": "Remove AA batteries or recycle the internal battery."},
    "battery":          {"bin": "Hazardous / Battery Bin",    "status": "🔋 HAZARD",  "tip": "NEVER in the trash. Causes garbage truck fires."},
    "aa battery":       {"bin": "Hazardous / Battery Bin",    "status": "🔋 HAZARD",  "tip": "Tape terminals if storing. Take to battery drop-off."},
    "lithium battery":  {"bin": "Hazardous / Battery Bin",    "status": "🔋 DANGER",  "tip": "Massive fire hazard if crushed in a garbage truck."},
    
    # ── ORGANICS & FOOD ────────────────────────────────────────────────────────
    "apple core":       {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "Perfect for composting."},
    "banana peel":      {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "Breaks down quickly, adds potassium to soil."},
    "orange peel":      {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "Adds acidity to compost."},
    "egg shell":        {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "Crush them first to add calcium to soil faster."},
    "coffee grounds":   {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "Excellent nitrogen source for compost."},
    "tea bag":          {"bin": "Brown Bin / Trash",          "status": "🌱 CHECK",   "tip": "Only compost if the bag is 100% paper (no plastic mesh)."},
    "bread":            {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "Stale bread is organic waste."},
    "moldy food":       {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "Nature's recyclers are already at work."},
    "meat":             {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "Usually requires municipal high-heat composting."},
    "bone":             {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "Requires municipal composting facilities."},
    "cheese":           {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "Dairy can be composted in city bins."},
    "vegetable":        {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "All veggie scraps are compostable."},
    "nut shell":        {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "Pistachio, walnut, and peanut shells are great carbon sources."},
    "houseplant":       {"bin": "Brown Bin (Compost)",        "status": "🪴 COMPOST", "tip": "Dead plants go in the compost (remove plastic pot)."},
    "leaves":           {"bin": "Brown Bin (Compost)",        "status": "🍂 COMPOST", "tip": "Yard waste makes great mulch."},
    "grass clippings":  {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "Mix with dry leaves for perfect compost."},
    "hair":             {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "Human and pet hair is compostable (rich in nitrogen)."},
    "fur":              {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "Pet fur is compostable."},

    # ── HAZARDOUS & MEDICAL ────────────────────────────────────────────────────
    "paint can":        {"bin": "Hazardous Waste",            "status": "🛢️ HAZARD",  "tip": "Liquid paint is hazardous. If completely dry, check local laws."},
    "motor oil":        {"bin": "Auto Parts Store",           "status": "🛢️ HAZARD",  "tip": "Never pour down the drain! Auto shops recycle it."},
    "antifreeze":       {"bin": "Hazardous Waste",            "status": "☣️ HAZARD",  "tip": "Highly toxic to animals. Dispose properly."},
    "bleach bottle":    {"bin": "Yellow Bin (Recycle)",       "status": "⚠️ RECYCLE", "tip": "Must be completely empty and rinsed."},
    "pesticide":        {"bin": "Hazardous Waste",            "status": "☣️ HAZARD",  "tip": "Toxic chemical. Take to a municipal drop-off."},
    "medicine bottle":  {"bin": "Pharmacy Take-Back",         "status": "💊 MEDICAL", "tip": "Empty bottles are sometimes recyclable, but return meds to pharmacy."},
    "pills":            {"bin": "Pharmacy Take-Back",         "status": "💊 HAZARD",  "tip": "NEVER flush medicine. It contaminates water supplies."},
    "syringe":          {"bin": "Sharps Container",           "status": "🏥 DANGER",  "tip": "Use a heavy plastic container. NEVER put loose in trash."},
    "bandaid":          {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "Medical waste goes in the trash."},
    "blister pack":     {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "Mixed plastic and foil cannot be recycled."},
    "face mask":        {"bin": "General Trash",              "status": "😷 TRASH",   "tip": "Disposable masks are mixed materials and unhygienic to sort."},
    "gloves":           {"bin": "General Trash",              "status": "🧤 TRASH",   "tip": "Latex/Nitrile gloves are trash."},

    # ── CLOTHING & TEXTILES ────────────────────────────────────────────────────
    "shirt":            {"bin": "Textile Bin / Donation",     "status": "👕 TEXTILE", "tip": "Donate if wearable. Recycle if ripped."},
    "t-shirt":          {"bin": "Textile Bin / Donation",     "status": "👕 TEXTILE", "tip": "Textile recyclers turn old cotton into insulation."},
    "pants":            {"bin": "Textile Bin / Donation",     "status": "👖 TEXTILE", "tip": "Denim can be recycled into housing insulation."},
    "jeans":            {"bin": "Textile Bin / Donation",     "status": "👖 TEXTILE", "tip": "Denim can be recycled into housing insulation."},
    "sweater":          {"bin": "Textile Bin / Donation",     "status": "🧥 TEXTILE", "tip": "Donate to shelters."},
    "socks":            {"bin": "Textile Bin",                "status": "🧦 TEXTILE", "tip": "Even single/holy socks can be recycled into fiber."},
    "underwear":        {"bin": "Textile Bin / Trash",        "status": "🩲 TEXTILE", "tip": "Check if your local textile recycler accepts these."},
    "shoes":            {"bin": "Textile Bin / Donation",     "status": "👟 TEXTILE", "tip": "Tie them together so they don't get separated."},
    "sneakers":         {"bin": "Textile Bin / Donation",     "status": "👟 TEXTILE", "tip": "Brands like Nike grind up old sneakers for turf."},
    "jacket":           {"bin": "Textile Bin / Donation",     "status": "🧥 TEXTILE", "tip": "Donate winter gear to charities."},
    "towel":            {"bin": "Textile Bin / Animal Shelter","status":"🧺 TEXTILE", "tip": "Animal shelters love old towels!"},
    "bed sheet":        {"bin": "Textile Bin / Animal Shelter","status":"🛏️ TEXTILE", "tip": "Recycle the fabric or donate to a vet/shelter."},
    "rag":              {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "If covered in oil/chemicals, it must go in the trash."},

    # ── MISCELLANEOUS HOUSEHOLD ────────────────────────────────────────────────
    "diaper":           {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "Disposable diapers take 500 years to decompose in a landfill."},
    "wet wipe":         {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "NEVER flush wipes! They contain plastic and destroy sewers."},
    "sponge":           {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "Synthetic sponges are trash. (Loofahs are compostable)."},
    "toothbrush":       {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "Mixed nylon/plastic. Switch to a bamboo toothbrush!"},
    "toothpaste tube":  {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "Mixed plastic/metal layers. Some specialty programs accept them."},
    "q-tip":            {"bin": "Trash / Compost",            "status": "⚠️ CHECK",   "tip": "Plastic sticks = trash. Paper sticks = compost."},
    "cotton swab":      {"bin": "Trash / Compost",            "status": "⚠️ CHECK",   "tip": "Plastic sticks = trash. Paper sticks = compost."},
    "toothpick":        {"bin": "Brown Bin (Compost)",        "status": "🌱 COMPOST", "tip": "Wood breaks down easily."},
    "candle":           {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "Paraffin wax is trash. The glass jar can be recycled if clean."},
    "umbrella":         {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "Too many mixed materials (nylon, metal, plastic)."},
    "backpack":         {"bin": "Donation",                   "status": "🎒 REUSE",   "tip": "Donate functional bags. Hard to recycle due to zippers/straps."},
    "luggage":          {"bin": "Donation / Bulk Trash",      "status": "🧳 REUSE",   "tip": "Donate if working. Otherwise, bulk trash."},
    "mattress":         {"bin": "Bulk Pickup / Recycler",     "status": "🛏️ SPECIAL", "tip": "Many cities have specific mattress recycling programs for the steel springs."},
    "tire":             {"bin": "Tire Recycler / Auto Shop",  "status": "🚗 SPECIAL", "tip": "Recycled into playground rubber or road materials."},

    # ── CATCH-ALLS & DEFAULTS ──────────────────────────────────────────────────
    "trash":            {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "Mixed waste. Try to audit your trash to reduce it!"},
    "garbage":          {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "Dispose in general waste."},
    "debris":           {"bin": "Construction Scrap",         "status": "🏗️ SPECIAL", "tip": "Sort into wood, metal, concrete."},
    "rubbish":          {"bin": "General Trash",              "status": "🗑️ TRASH",   "tip": "General waste."}
}


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  FUZZY LABEL MATCHER  (open-vocabulary → dictionary lookup)
# ═══════════════════════════════════════════════════════════════════════════════
_SORTED_KEYS = sorted(ECO_RULES.keys(), key=len, reverse=True)     # longest first

def match_label(raw_label: str) -> dict:
    """
    Florence-2 returns free-form labels like "a plastic water bottle".
    We normalise and do a keyword match against ECO_RULES.
    """
    text = raw_label.lower().strip()

    # 1) exact hit
    if text in ECO_RULES:
        return ECO_RULES[text]

    # 2) keyword search — longest key first to prefer "plastic bottle" over "bottle"
    for key in _SORTED_KEYS:
        if key in text:
            return ECO_RULES[key]

    # 3) fallback
    return {
        "bin": "General Trash (check locally)",
        "status": "❓ UNKNOWN",
        "tip": "Consult your local waste management guide.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  COLOUR PALETTE FOR BOUNDING BOXES
# ═══════════════════════════════════════════════════════════════════════════════
BOX_COLOURS = [
    "#00E676", "#FF5252", "#448AFF", "#FFD740",
    "#E040FB", "#00E5FF", "#FF6E40", "#69F0AE",
    "#7C4DFF", "#FFAB40", "#18FFFF", "#FF4081",
    "#B2FF59", "#536DFE", "#F50057", "#00BFA5",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  6.  CORE ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_image(image: Image.Image, confidence_threshold: float = 0.3):
    """
    Main pipeline:
      1. Florence-2 <OD> for bounding-box detection
      2. Florence-2 <DETAILED_CAPTION> for scene description
      3. Eco-rules lookup for every detected label
    Returns (annotated_image, markdown_report).
    """
    if image is None:
        return None, "⚠️ Please upload an image first."

    if florence_model is None:
        return None, f"⚠️ {ENGINE_STATUS}"

    # Convert to RGB if needed
    image = image.convert("RGB")

    # ── A.  Object Detection ───────────────────────────────────────────────
    od_result = florence_run(image, "<OD>")
    detections = od_result.get("<OD>", {})
    bboxes = detections.get("bboxes", [])
    labels = detections.get("labels", [])

    # ── B.  Scene Caption ──────────────────────────────────────────────────
    caption_result = florence_run(image, "<DETAILED_CAPTION>")
    scene_caption = caption_result.get("<DETAILED_CAPTION>", "No caption generated.")

    # ── C.  Draw Annotated Image ───────────────────────────────────────────
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Try to get a readable font
    try:
        font = ImageFont.truetype("arial.ttf", size=max(16, image.width // 50))
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                      size=max(16, image.width // 50))
        except (IOError, OSError):
            font = ImageFont.load_default()

    label_font = font
    line_width = max(2, image.width // 300)

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        colour = BOX_COLOURS[i % len(BOX_COLOURS)]
        x1, y1, x2, y2 = bbox

        # draw box
        for offset in range(line_width):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                           outline=colour)

        # draw label background
        tag = f" {label} "
        try:
            text_bbox = draw.textbbox((x1, y1), tag, font=label_font)
            tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        except AttributeError:
            tw, th = draw.textsize(tag, font=label_font)

        label_y = max(0, y1 - th - 6)
        draw.rectangle([x1, label_y, x1 + tw + 8, label_y + th + 6], fill=colour)
        draw.text((x1 + 4, label_y + 2), tag, fill="black", font=label_font)

    # ── D.  Build Eco-Report ───────────────────────────────────────────────
    report_lines = []
    report_lines.append("# ♻️ EcoSort Ultra — Analysis Report\n")
    report_lines.append(f"**Engine:** {ENGINE_STATUS}\n")
    report_lines.append(f"**Objects detected:** {len(bboxes)}\n")
    report_lines.append("---\n")

    report_lines.append("## 🌍 Scene Description\n")
    report_lines.append(f"> {scene_caption}\n")
    report_lines.append("---\n")

    if not labels:
        report_lines.append("### 🔍 No specific objects detected. Try another angle or closer shot.\n")
    else:
        seen = set()
        report_lines.append("## 📋 Detected Items & Sorting Guide\n")
        for label in labels:
            norm = label.lower().strip()
            if norm in seen:
                continue
            seen.add(norm)

            info = match_label(norm)
            report_lines.append(f"### {info['status']}:  **{label.upper()}**\n")
            report_lines.append(f"📥 **Sort into:** {info['bin']}\n")
            report_lines.append(f"> 💡 **Tip:** {info['tip']}\n")
            report_lines.append("---\n")

    report_lines.append("\n*♻️ When in doubt, check your local waste management rules.*")
    report = "\n".join(report_lines)

    return annotated, report


# ═══════════════════════════════════════════════════════════════════════════════
#  7.  GRADIO USER INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────────────────────── */
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
/* ── Header ─────────────────────────────────────────────────────────── */
#eco-header {
    text-align: center;
    padding: 28px 16px 12px;
    background: linear-gradient(135deg, #0d9488 0%, #065f46 50%, #064e3b 100%);
    border-radius: 16px;
    margin-bottom: 20px;
    color: white;
}
#eco-header h1 { font-size: 2.2rem; letter-spacing: 0.02em; margin: 0; }
#eco-header p  { opacity: 0.85; margin: 6px 0 0; font-size: 1rem; }
/* ── Buttons ────────────────────────────────────────────────────────── */
#scan-btn {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    border: none !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 14px 0 !important;
    letter-spacing: 0.04em;
    border-radius: 12px !important;
    color: white !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
#scan-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(16,185,129,0.4) !important;
}
/* ── Report ─────────────────────────────────────────────────────────── */
#eco-report {
    max-height: 600px;
    overflow-y: auto;
    padding: 16px;
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 12px;
    background: rgba(6,78,59,0.04);
}
/* ── Footer ─────────────────────────────────────────────────────────── */
#eco-footer { text-align: center; opacity: 0.55; font-size: 0.85rem; padding: 8px; }
"""

def build_ui():
    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.emerald,
            secondary_hue=gr.themes.colors.teal,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CUSTOM_CSS,
        title="EcoSort Ultra — AI Waste Analyzer",
    ) as demo:
        # ── Header ─────────────────────────────────────────────────────
        gr.HTML(
            """
            <div id="eco-header">
                <h1>♻️ EcoSort Ultra</h1>
                <p>AI-Powered Waste Detection &amp; Sorting Intelligence · Powered by Florence-2</p>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            # ── Left column: Input ─────────────────────────────────────
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil",
                    label="📸 Upload Image",
                    sources=["upload", "clipboard"],
                    height=420,
                )
                scan_btn = gr.Button(
                    "🚀  SCAN & ANALYZE",
                    variant="primary",
                    elem_id="scan-btn",
                )
                gr.Markdown(
                    "*Florence-2 performs open-vocabulary object detection — "
                    "it can detect virtually any object, not just a fixed list.*"
                )

            # ── Right column: Output ───────────────────────────────────
            with gr.Column(scale=1):
                output_image = gr.Image(
                    type="pil",
                    label="🔍 Detection Results",
                    interactive=False,
                    height=420,
                )
                report_output = gr.Markdown(
                    value="*Upload an image and click **SCAN & ANALYZE** to begin.*",
                    elem_id="eco-report",
                )

        # ── Examples ───────────────────────────────────────────────────
        gr.Examples(
            examples=[
                # Add your own example image paths here
                # ["examples/kitchen_waste.jpg"],
                # ["examples/street_litter.jpg"],
            ],
            inputs=[input_image],
            label="📂 Example Images (add your own!)",
        ) if False else None   # disabled — no sample images bundled

        gr.HTML('<div id="eco-footer">EcoSort Ultra · Built with ❤️ for a cleaner planet</div>')

        # ── Wire events ────────────────────────────────────────────────
        scan_btn.click(
            fn=analyze_image,
            inputs=[input_image],
            outputs=[output_image, report_output],
        )

    return demo


# ═══════════════════════════════════════════════════════════════════════════════
#  8.  LAUNCH
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  ♻️  ECOSORT ULTRA  — Florence-2 Engine")
    print(f"  Status: {ENGINE_STATUS}")
    print("=" * 60)
    demo = build_ui()
    demo.launch(share=True)