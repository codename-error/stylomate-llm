# LLM Service documentation

# Images Feature Extraction

## Overview

Feature extraction of clothing is an automated process to identify and extract attributes from clothing items in images, such as category (top/bottom), type (sweatshirt, skirt), color, pattern (solid, stripes), and length (hip, ankle), which are then represented in a structured format such as JSON; this process is important for data organization, search, recommendation, and fashion trend analysis.

## Feature

Attribute Extraction Guidelines :

1. category (Clothing Category)

Indicate whether the item is a Top or a Bottom

Use one of the following: Top, Bottom

1. type (Type of Item)

Identify the general category of the bottom item. Use common and well-understood fashion terms

Example values: Jeans, Trousers, Joggers, Shorts, Skirt, Culottes, Leggings, Chinos, Cargo Pants, Maxi Skirt, Mini Skirt

1. color (Color of the Item)

Identify the most conspicuous or primary color. If there are two strong colors, mention both of them separated by "and" (e.g., Black and White). If there is an accent color with prominent significance, mention it in the second place (e.g., Navy Blue with Yellow Accents)

Mention regular color names: Red, Light Blue, Army Green, Gray, Ivory White, Beige, Dark Purple, etc.

1. pattern (Fabric Pattern)

If no pattern is visible, mention: Solid

Sample values: Solid, Vertical Stripes, Horizontal Stripes, Tartan Plaid, Polkadots, Floral, Abstract, Geometric, Tie-dye, Animal Print, Batik

1. length (Item Length)

Define in terms of how far down the body the item falls

Example values:

- Short – above mid-thigh
- Mid-Thigh – mid-thigh level
- Knee – around knee level
- Calf – mid-calf
- Ankle – full-length to ankle
- Floor-Length – touches or covers the feet

## Workflow

### Instalation in Colab

```python
!pip install python-dotenv langchain langchain-google-genai
```

### API configuration

```python
import os
os.environ["GEMINI_API_KEY"] = "your_api_key_here"
```

### Model Setup

```python
from langchain_google_genai import ChatGoogleGenerativeAI

def setup_gemini_chat():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY tidak ditemukan")
    
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.5
    )

chat = setup_gemini_chat()
```

### clothes and Pants input

```python
prompt = f"""
Analyze this clothing item image and provide detailed attributes in EXACTLY this JSON format:*
{
  "category": "Top/Bottom/Dress/Outerwear",
  "type": "T-Shirt/Blouse/Shirt/Sweater/Jeans/Skirt/Shorts/etc",
  "color": "Standard color name (e.g., Ivory White, Navy Blue, Jet Black)",
  "pattern": "Plain/Striped/Checkered/Floral/Graphic/Polka Dot/Abstract/etc",
  "length": "Crop/Waist/Hip/Midi/Maxi (for tops) or Short/Knee-Length/Ankle-Length (for bottoms)"
}

*Requirements:*
1. Be strictly consistent with the JSON structure and field names
2. Use only the specified values for each field:
   - Category: Top, Bottom, Dress, or Outerwear
   - Type: Specific garment type matching the category
   - Color: Standard color names (avoid vague descriptions)
   - Pattern: Clear pattern description or "Plain" if no pattern
   - Length: Appropriate length measurement for the category
3. If any attribute cannot be determined, use "Unknown"
4. Focus on the main clothing item in the image
5. Output ONLY raw JSON without markdown formatting
6. Ensure valid JSON syntax (double quotes, proper commas)

*Example valid response:*
{
  "category": "Top",
  "type": "Blouse",
  "color": "Ivory White",
  "pattern": "Small Flowers",
  "length": "Hip"
}
"""
response = chat.invoke(prompt)
print(response.content)
```

Example output

```python
{
  "category": "Bottom",
  "type": "Culottes",
  "color": "Beige",
  "pattern": "Solid",
  "length": "Calf"
}
```

# Images Recommendation System

## Overview

This system is a system that can provide outfit combination recommendations (top & bottom) based on Generative AI. This system utilizes Google's Gemini 1.5 Flash model through integration with LangChain. The goal of this system is to help users choose the best outfit combination based on their style preferences, colors, as well as the context of their activities.

## Feature

- **`style_preference`**: Indicates the user's preferred fashion style. Examples include `"casual"`, `"formal"`, `"sporty"`, etc. This helps the AI prioritize outfit types that match the user's personal taste.
- **`time_stamp`**: The exact timestamp when the request is made or when the outfit is needed. This can be used for future enhancements like time-aware styling (e.g., day vs. night).
- **`Activity`**: Describes what the user will be doing (e.g., `"watch a football in the stadium"`). This guides the AI to recommend contextually appropriate outfits.

## Workflow

### Instalation in Colab

```python
!pip install python-dotenv langchain langchain-google-genai
```

### API configuration

```python
import os
os.environ["GEMINI_API_KEY"] = "your_api_key_here"
```

### Model Setup

```python
from langchain_google_genai import ChatGoogleGenerativeAI

def setup_gemini_chat():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY tidak ditemukan")
    
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.5
    )

chat = setup_gemini_chat()
```

### Data Input (Sample)

Tops

```python
Tops = [
    {"id": "T001", "category": "Top", "type": "Blouse", "color": "Sky Blue", "pattern": "Plain", "length": "Hip"},
    {"id": "T003", "category": "Top", "type": "Shirt", "color": "Ivory White", "pattern": "Striped", "length": "Hip"},
    {"id": "T004", "category": "Top", "type": "Shirt", "color": "White", "pattern": "Batik", "length": "Hip"},
    {"id": "T005", "category": "Top", "type": "Blouse", "color": "Ivory White", "pattern": "Small Flowers", "length": "Hip"},
    {"id": "T006", "category": "Top", "type": "Shirt", "color": "White", "pattern": "Solid", "length": "Hip"},
    {"id": "T007", "category": "Top", "type": "Hoodie", "color": "White and Black", "pattern": "Graphic", "length": "Hip"},
    {"id": "T008", "category": "Top", "type": "Shirt", "color": "White", "pattern": "Solid", "length": "Hip"},
    {"id": "T009", "category": "Top", "type": "T-shirt", "color": "Red", "pattern": "Solid", "length": "Waist"},
    {"id": "T010", "category": "Top", "type": "Shirt", "color": "Brown and Beige", "pattern": "Batik", "length": "Hip"},
    {"id": "T011", "category": "Top", "type": "Shirt", "color": "White", "pattern": "Solid", "length": "Hip"}
]
```

Bottoms

```python
Bottoms = [
    {"id": "B001", "category": "Bottom", "type": "Jeans", "color": "Light Blue", "pattern": "Plain", "length": "Ankle-Length"},
    {"id": "B002", "category": "Bottom", "type": "Skirt", "color": "Beige", "pattern": "Plain", "length": "Knee-Length"},
    {"id": "B003", "category": "Bottom", "type": "Shorts", "color": "Jet Black", "pattern": "Plain", "length": "Short"},
    {"id": "B004", "category": "Bottom", "type": "Culottes", "color": "Beige", "pattern": "Solid", "length": "Calf"},
    {"id": "B005", "category": "Bottom", "type": "Cargo Pants", "color": "Black", "pattern": "Solid", "length": "Floor-Length"},
    {"id": "B006", "category": "Bottom", "type": "Trousers", "color": "Black", "pattern": "Solid", "length": "Ankle"},
    {"id": "B007", "category": "Bottom", "type": "Trousers", "color": "Black", "pattern": "Solid", "length": "Ankle"},
    {"id": "B008", "category": "Bottom", "type": "Chinos", "color": "Tan", "pattern": "Solid", "length": "Ankle"}
]
```

### User Preference and Context

```python
user_preference = [
    {
        "style_preference": "casual",  
    }
]
Context = [
    {
        "time_stamp": "2025-05-03 01:38:52.002722",
        "Activity": "watch a football in the stadium",
    }
]
```

### Outfit Recommendation

1. Top (clothes) Recommendation

if we need a shirt recommendation based on the inputted pants

```python
fashion_prompt = f"""
You are an AI fashion stylist. Given a bottom item and a list of available top items (with unique IDs), recommend the most suitable top by returning only the ID of the best-matching item.
Input Format:
Bottom item:
 {
 "category": "Bottom",
 "type": "Skirt",
 "color": "Beige",
 "pattern": "Plain",
 "length": "Knee-Length"
 }
Available tops:
 {Tops}
 
User Preferences:
 {user_preference}
Context:
 {Context}
 
Output Format (only valid JSON, no markdown):
 {
 "recommended_top_id": "T003"
 }
Guidelines:
1. Choose the best-matching top based on type, color, pattern, and length harmony with the bottom.
2. Prioritize alignment with style_preference (must reflect the user’s fashion style).
3. Consider the occasion and color preference as secondary factors.
4. Do not include explanation—only return the ID.
5. Output must be valid JSON with double quotes.
"""
response = chat.invoke(fashion_prompt)
print(response.content)
```

Example Output

```python
{
 "recommended_top_id": "T004"
}
```

1. Bottom (pants) Recommendation

if we need pants recommendations based on the inputted clothes

```python
fashion_prompt = f"""
You are an AI fashion stylist. Given a top item and a list of available bottom items (with unique IDs), recommend the most suitable bottom by returning only the ID of the best-matching item.
Input Format:
Top item:
 {
 "category": "Top",
 "type": "Blouse",
 "color": "Ivory White",
 "pattern": "Small Flowers",
 "length": "Hip"
 }
Available bottoms:
{Bottoms}

User Preferences: 
{user_preference}

Context:
{Context}

Output Format (only valid JSON, no markdown):
 {
 "recommended_bottom_id": "B002"
 }
Guidelines:
1. Choose the best-matching bottom based on type, color, pattern, and length harmony with the top. 
2. Alignment with style_preference (highest priority — must reflect the user’s fashion style)
3. Consider the occasion and color preference as secondary factors.
4. Do not include explanation—only return the ID.
5. Use only the id field from the matching item as output.
6. Output must be valid JSON with double quotes.
"""
response = chat.invoke(fashion_prompt)
print(response.content)
```

Example Output

```python
{
 "recommended_bottom_id": "B003"
}
```

1. outfit combination recommendation

if we need two types of recommendations at once

```python
fashion_prompt = f"""
You are an AI fashion stylist. Given a list of top items and a list of bottom items (each with unique IDs), recommend the best **combination** (top + bottom) that matches well based on style, color, pattern, length harmony, user preferences, and context.

Available tops:
{Tops}

Available bottoms:
{Bottoms}

User Preferences:
{user_preference}

Context:
{Context}                         

Output Format (only valid JSON, no markdown): 
For example 
{
  "recommended_combination": {
    "top_id": "T003",
    "bottom_id": "B002"
  }
}

Guidelines:
1. Evaluate all possible top-bottom pairings.
2. Prioritize alignment with `style_preference` (must match user’s style).
3. Ensure visual harmony in color, pattern, and length.
4. Consider suitability for the given `occasion`.
5. Return only the best combination.
6. Do not include explanation—only output the selected IDs.
7. Output must be valid JSON with double quotes.
"""

response = chat.invoke(fashion_prompt)
print(response.content)
```

Example Output

```python
{
  "recommended_combination": {
    "top_id": "T005",
    "bottom_id": "B002"
  }
}
```
