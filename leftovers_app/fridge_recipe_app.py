
import streamlit as st
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass, asdict

# Imaging
from PIL import Image
import io
import base64

# Barcode (optional)
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from pyzbar import pyzbar
    HAS_PYZBAR = True
except Exception:
    HAS_PYZBAR = False

# OpenAI (for image vision + fallback JSON forcing)
from openai import OpenAI

# LangChain / OpenAI wrappers
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser

# RAG engine
from rag_engine import generate_recipes_rag

# ---------------- Config ----------------
CONFIG_PATH = "config.json"
cfg = json.load(open(CONFIG_PATH, "r", encoding="utf-8")) if os.path.exists(CONFIG_PATH) else {}

APP_TITLE = cfg.get("app_config", {}).get("title", "AI Recipe Generator")
EMBED_MODEL = cfg.get("ai_config", {}).get("embedding_model", "text-embedding-3-small")
LLM_MODEL = cfg.get("ai_config", {}).get("model", "gpt-4o-mini")
VECTOR_STORE_PATH = cfg.get("storage", {}).get("vector_store_path", "recipe_vector_store")

# ---------------- Data classes ----------------
@dataclass
class Ingredient:
    name: str
    quantity: str = ""
    category: str = ""
    is_leftover: bool = False
    expiry_status: str = "fresh"  # fresh, soon, expired

@dataclass
class Recipe:
    name: str
    ingredients_needed: List[str]
    ingredients_available: List[str]
    missing_ingredients: List[str]
    instructions: List[str]
    cooking_time: str
    difficulty: str
    calories: str = ""
    suitable_for_leftovers: bool = False

# ---------------- Utilities ----------------
def _image_to_data_url(img_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _guess_mime(fname: Optional[str]) -> str:
    if not fname: return "image/png"
    name = fname.lower()
    if name.endswith(".jpg") or name.endswith(".jpeg"):
        return "image/jpeg"
    if name.endswith(".webp"):
        return "image/webp"
    if name.endswith(".png"):
        return "image/png"
    return "image/png"

# ---------------- Generator ----------------
class RecipeGenerator:
    def __init__(self, api_key: str):
        # LangChain chat model (JSON enforced)
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=cfg.get("ai_config", {}).get("temperature", 0.7),
            api_key=api_key,
            model_kwargs={ "response_format": {"type": "json_object"} }
        )
        # Embeddings for FAISS
        self.embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=api_key)
        # Native OpenAI client (vision)
        self.oa_client = OpenAI(api_key=api_key)

        self.vector_store = None
        self.load_or_create_vector_store()

    def load_or_create_vector_store(self):
        # Try to load FAISS; if not exists, build from local JSON (recipe_database.json) minimal seed
        try:
            from langchain_community.vectorstores import FAISS
        except Exception:
            self.vector_store = None
            return

        if os.path.isdir(VECTOR_STORE_PATH):
            try:
                self.vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH, self.embeddings, allow_dangerous_deserialization=True
                )
                return
            except Exception:
                pass

        # Build minimal vector store from recipe_database.json if available
        docs = []
        if os.path.exists("recipe_database.json"):
            data = json.load(open("recipe_database.json", "r", encoding="utf-8"))
            for r in data.get("recipes", []):
                text = (
                    f"TITLE: {r.get('name','')}\n"
                    f"DESC: {r.get('description','')}\n"
                    f"INGREDIENTS: {', '.join(r.get('ingredients', []))}\n"
                    f"TAGS: {', '.join(r.get('tags', []))}\n"
                    f"LEFTOVER: {r.get('suitable_for_leftovers', False)}\n"
                )
                docs.append(Document(page_content=text, metadata={"title": r.get("name","")}))
        else:
            # Fallback seed
            sample_recipes = [
                "김치볶음밥: 김치, 밥, 계란, 파, 참기름",
                "피자 프리타타: 남은 피자, 계란, 우유, 치즈",
                "라면 볶음밥: 남은 라면, 밥, 계란, 김치"
            ]
            docs = [Document(page_content=t) for t in sample_recipes]

        from langchain_community.vectorstores import FAISS
        vs = FAISS.from_documents(docs, self.embeddings)
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        vs.save_local(VECTOR_STORE_PATH)
        self.vector_store = vs

    # ---------- Image analysis via Vision LLM ----------
    def analyze_image_for_ingredients(self, image_bytes: bytes, filename: Optional[str] = None) -> List[str]:
        # Use OpenAI chat completions with image
        mime = _guess_mime(filename or "")
        data_url = _image_to_data_url(image_bytes, mime)
        prompt_text = (
            "Identify all food items, ingredients, and leftover dishes visible in this image. "
            "If you see cooked dishes (e.g., pizza slice, fried chicken, tteokbokki), include them as 'leftover ...'. "
            "Return a JSON array of short ingredient/leftover names (lowercase)."
        )
        resp = self.oa_client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            response_format={ "type": "json_object" },
            messages=[
                {"role":"system","content":"You are an expert at identifying food items from images."},
                {"role":"user","content":[
                    {"type":"text","text": prompt_text},
                    {"type":"image_url","image_url":{"url": data_url}}
                ]}
            ]
        )
        try:
            content = resp.choices[0].message.content
            data = json.loads(content)
            if isinstance(data, dict):
                for k in ["ingredients", "items", "list"]:
                    if k in data and isinstance(data[k], list):
                        return [str(x) for x in data[k]]
                flat = []
                for v in data.values():
                    if isinstance(v, list):
                        flat.extend(v)
                return [str(x) for x in flat]
            elif isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass
        return []

    # ---------- Barcode ----------
    def scan_barcode(self, image_data: bytes) -> Optional[str]:
        try:
            if HAS_PYZBAR and HAS_CV2:
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                codes = pyzbar.decode(image)
                if codes:
                    return codes[0].data.decode("utf-8", errors="ignore")
            if HAS_CV2 and hasattr(cv2, "barcode_BarcodeDetector"):
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                det = cv2.barcode_BarcodeDetector()
                ok, decoded_info, decoded_type, points = det.detectAndDecode(img)
                if ok and decoded_info:
                    return decoded_info[0]
        except Exception:
            return None
        return None

    # ---------- Categorize ----------
    def categorize_ingredients(self, ingredients: List[str]) -> List[Ingredient]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Categorize the following ingredients and identify if they are leftovers.
Categories: vegetables, proteins, dairy, grains, condiments, leftovers, others.
Return strictly as JSON array like:
[{"name":"ingredient","category":"category","is_leftover":true}]"""),
            ("user", f"Categorize these: {', '.join(ingredients)}")
        ])
        chain = prompt | self.llm | JsonOutputParser()
        try:
            result = chain.invoke({})
            return [Ingredient(**item) for item in result]
        except Exception:
            return [Ingredient(name=x, category=("leftovers" if "leftover" in x.lower() else "others"),
                               is_leftover=("leftover" in x.lower())) for x in ingredients]

    # ---------- Recommend ----------
    def recommend_recipes(self, ingredients: List[Ingredient], use_rag: bool = True) -> List[Dict]:
        ingredient_names = [ing.name for ing in ingredients]
        has_leftovers = any(ing.is_leftover for ing in ingredients)

        if use_rag:
            structured = {
                "ingredients": [{"name": n, "quantity": None, "state": None, "is_leftover": ("leftover" in n.lower())} for n in ingredient_names],
                "leftover_dishes": [n for n in ingredient_names if "leftover" in n.lower()],
                "notes": [], "language": "ko"
            }
            rag = generate_recipes_rag(structured, model=LLM_MODEL, k=5)
            ui = []
            for r in rag.get("recipes", []):
                ui.append({
                    "name": r.get("title", "Recipe"),
                    "description": "Uses: " + ", ".join(r.get("uses", [])),
                    "uses_leftovers": any("leftover" in u for u in r.get("uses", [])),
                    "difficulty": "medium",
                    "time": f"{r.get('time_minutes','~30')} min"
                })
            if ui:
                return ui

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a creative chef specializing in using leftover foods.
Return 3-5 recipes as JSON list with keys: name, description, uses_leftovers (bool), difficulty (easy/medium/hard), time."""),
            ("user", f"Available ingredients: {', '.join(ingredient_names)}\nHas leftovers: {has_leftovers}")
        ])
        chain = prompt | self.llm | JsonOutputParser()
        return chain.invoke({})

    # ---------- Details ----------
    def generate_recipe_details(self, recipe_name: str, available_ingredients: List[str]) -> Recipe:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate a detailed recipe with step-by-step instructions.
Return JSON with keys: name, ingredients_needed (list), instructions (list), cooking_time, difficulty, calories, suitable_for_leftovers (bool)."""),
            ("user", f"Recipe: {recipe_name}\nAvailable ingredients: {', '.join(available_ingredients)}")
        ])
        chain = prompt | self.llm | JsonOutputParser()
        data = chain.invoke({})

        need = data.get("ingredients_needed", [])
        avail = [a.lower() for a in available_ingredients]
        missing = [x for x in need if not any(a in x.lower() or x.lower() in a for a in avail)]

        return Recipe(
            name=data.get("name", recipe_name),
            ingredients_needed=need,
            ingredients_available=available_ingredients,
            missing_ingredients=missing,
            instructions=data.get("instructions", []),
            cooking_time=data.get("cooking_time", "~30 minutes"),
            difficulty=data.get("difficulty", "medium"),
            calories=data.get("calories", ""),
            suitable_for_leftovers=data.get("suitable_for_leftovers", True)
        )

    def suggest_substitutions(self, missing_ingredients: List[str], available_ingredients: List[str]) -> Dict[str, str]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Suggest substitutions for missing ingredients using available ones.
Return JSON object where keys are missing ingredients and values are suggested substitutions."""),
            ("user", f"Missing: {', '.join(missing_ingredients)}\nAvailable: {', '.join(available_ingredients)}")
        ])
        chain = prompt | self.llm | JsonOutputParser()
        return chain.invoke({})

# ---------------- Streamlit App ----------------
st.set_page_config(page_title=APP_TITLE, page_icon="🥘", layout="wide")
st.title(APP_TITLE)
st.markdown("*Transform your leftovers into delicious meals!*")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    use_rag = st.checkbox("Use FAISS retrieval (RAG)", value=True)
    if not api_key:
        st.warning("Enter your OpenAI API key to continue")
        st.stop()
    st.success("API key set")
    generator = RecipeGenerator(api_key)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Input Ingredients", "🍽️ Recipe Recommendations", "👨‍🍳 Recipe Details", "📊 History"])

with tab1:
    st.header("Add Your Ingredients")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📝 Text Input")
        text_input = st.text_area("Enter ingredients (one per line or comma-separated)",
                                  height=150,
                                  placeholder="e.g., leftover pizza, kimchi, rice, eggs...")
        if st.button("Add Text Ingredients", type="primary"):
            if text_input:
                new_ings = [x.strip() for x in text_input.replace("\n", ",").split(",") if x.strip()]
                categorized = generator.categorize_ingredients(new_ings)
                if "ingredients" not in st.session_state:
                    st.session_state.ingredients = []
                st.session_state.ingredients.extend(categorized)
                st.success(f"Added {len(new_ings)} ingredients!")
                st.rerun()

    with col2:
        st.subheader("📸 Image Input")
        up = st.file_uploader("Upload a photo of your fridge/leftovers",
                              type=["png", "jpg", "jpeg", "webp"])
        if up is not None:
            st.image(Image.open(up), caption="Uploaded", use_column_width=True)
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing image (LLM Vision)..."):
                    items = generator.analyze_image_for_ingredients(up.getvalue(), filename=getattr(up, "name", ""))
                    if items:
                        categorized = generator.categorize_ingredients(items)
                        st.session_state.ingredients.extend(categorized)
                        st.success(f"Detected {len(items)} items")
                        st.rerun()
                    else:
                        st.error("No items detected. Try a clearer photo.")

    with col3:
        st.subheader("📊 Barcode Scanner")
        bc = st.file_uploader("Upload barcode image", type=["png","jpg","jpeg","webp"])
        if bc is not None:
            st.image(Image.open(bc), caption="Barcode Image", use_column_width=True)
            if st.button("Scan Barcode", type="primary"):
                code = generator.scan_barcode(bc.getvalue())
                if code:
                    st.success(f"Scanned: {code}")
                    categorized = generator.categorize_ingredients([f"product {code}"])
                    st.session_state.ingredients.extend(categorized)
                    st.rerun()
                else:
                    st.error("Could not scan barcode")

    if "ingredients" in st.session_state and st.session_state["ingredients"]:
        st.divider()
        st.subheader("📦 Current Ingredients")
        cats = {}
        for ing in st.session_state.ingredients:
            cats.setdefault(ing.category or "others", []).append(ing)
        cols = st.columns(max(1, len(cats)))
        for i, (cat, items) in enumerate(cats.items()):
            with cols[i]:
                st.markdown(f"**{cat.title()}**")
                for it in items:
                    prefix = "♻️ " if it.is_leftover else "• "
                    st.markdown(f"{prefix}{it.name}")
    else:
        st.info("Add ingredients via text, image, or barcode.")

with tab2:
    st.header("Recipe Recommendations")
    if "ingredients" not in st.session_state or not st.session_state.ingredients:
        st.info("👈 Please add ingredients first in the 'Input Ingredients' tab")
    else:
        if st.button("🎯 Get Recipe Recommendations", type="primary"):
            with st.spinner("Finding recipes..."):
                recs = generator.recommend_recipes(st.session_state.ingredients, use_rag=use_rag)
                st.session_state.recipes = recs

        if "recipes" in st.session_state and st.session_state.recipes:
            st.subheader("Recommended Recipes")
            for idx, recipe in enumerate(st.session_state.recipes):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### {idx+1}. {recipe.get('name','(untitled)')}")
                    st.markdown(f"*{recipe.get('description','')}*")
                    tags = []
                    if recipe.get("uses_leftovers"):
                        tags.append("♻️ Uses Leftovers")
                    tags.append(f"⏱️ {recipe.get('time','~30 min')}")
                    tags.append(f"📊 {recipe.get('difficulty','medium').title()}")
                    st.markdown(" | ".join(tags))
                with col2:
                    if st.button("View Recipe", key=f"view_{idx}"):
                        st.session_state.selected_recipe = recipe.get("name")
                        st.rerun()
                st.divider()

with tab3:
    st.header("Recipe Details & Instructions")
    if st.session_state.get("selected_recipe"):
        with st.spinner(f"Generating recipe for {st.session_state['selected_recipe']}..."):
            available = [ing.name for ing in st.session_state.get("ingredients", [])]
            recipe = generator.generate_recipe_details(st.session_state["selected_recipe"], available)
            st.markdown(f"# {recipe.name}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cooking Time", recipe.cooking_time)
            c2.metric("Difficulty", recipe.difficulty.title())
            c3.metric("Calories", recipe.calories or "~500 kcal")
            if recipe.suitable_for_leftovers:
                c4.success("♻️ Leftovers Friendly")

            st.divider()
            left, right = st.columns(2)
            with left:
                st.subheader("✅ Available Ingredients")
                for ing in recipe.ingredients_available:
                    st.markdown(f"• {ing}")
            with right:
                st.subheader("🛒 Shopping List")
                if recipe.missing_ingredients:
                    for ing in recipe.missing_ingredients:
                        st.markdown(f"• ~~{ing}~~")
                else:
                    st.success("You have all ingredients!")

            st.divider()
            st.subheader("👨‍🍳 Cooking Instructions")
            for i, step in enumerate(recipe.instructions, 1):
                st.markdown(f"**Step {i}:** {step}")

            st.divider()
            if st.button("💾 Save Recipe", type="primary"):
                if "recipe_history" not in st.session_state:
                    st.session_state.recipe_history = []
                st.session_state.recipe_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "recipe": asdict(recipe)
                })
                st.success("Recipe saved to history!")
    else:
        st.info("👈 Select a recipe from the 'Recipe Recommendations' tab")

with tab4:
    st.header("Recipe History")
    for entry in reversed(st.session_state.get("recipe_history", [])):
        ts = entry["timestamp"]
        data = entry["recipe"]
        with st.expander(f"{data['name']} - {ts[:16].replace('T',' ')}"):
            st.markdown(f"**Cooking Time:** {data['cooking_time']}")
            st.markdown(f"**Difficulty:** {data['difficulty']}")
            st.markdown("**Instructions:**")
            for i, step in enumerate(data['instructions'], 1):
                st.markdown(f"{i}. {step}")
