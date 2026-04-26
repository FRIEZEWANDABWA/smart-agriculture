from typing import Dict, TypedDict

class Advice(TypedDict):
    overview: str
    action: str
    chemical: str
    organic: str

def get_recommendation(disease_class: str) -> Advice:
    """Returns a highly detailed, commercial-grade agronomic profile based on the disease."""
    
    database: Dict[str, Advice] = {
        "Blight": {
            "overview": "Northern Corn Leaf Blight (NCLB) is a fungal disease characterized by large, cigar-shaped necrotic lesions. It causes catastrophic yield loss if it infects the plant before the silking stage.",
            "action": "Immediate Action: Scout the entire field. If lesions are present on or above the ear leaf just prior to tasseling, economic threshold for spraying is met.",
            "chemical": "Chemical Intervention: Apply foliar fungicides containing active ingredients like Pyraclostrobin, Azoxystrobin, or Propiconazole within 5-7 days.",
            "organic": "Cultural/Organic Practice: Deep-till infected residue post-harvest. Practice 1-2 year crop rotation with soybeans or non-host legumes."
        },
        "Common_Rust": {
            "overview": "Common Rust is driven by the fungus *Puccinia sorghi*. It produces brick-red, raised pustules on the upper and lower leaf surfaces, heavily draining the plant's carbohydrate reserves.",
            "action": "Immediate Action: Isolate and monitor. Rust blows in on weather fronts, making it highly dependent on cool, humid conditions.",
            "chemical": "Chemical Intervention: Spray a systemic Triazole or Strobilurin-based fungicide if the infection occurs early in the vegetative phases.",
            "organic": "Cultural/Organic Practice: Plant highly resistant commercial hybrids."
        },
        "Gray_Leaf_Spot": {
            "overview": "Gray Leaf Spot (GLS) is a major fungal threat thriving in high humidity. Recognizable by its tan, perfectly rectangular lesions restricted by the leaf veins.",
            "action": "Immediate Action: Assess the lower canopy. GLS starts from the ground up from previous crop residue. Do not allow it to reach the ear leaf.",
            "chemical": "Chemical Intervention: Deploy preventative spray programs using heavy strobilurins if weather forecasted is warm and highly overcast.",
            "organic": "Cultural/Organic Practice: Limit early season overhead irrigation to reduce canopy moisture."
        },
        "Healthy": {
            "overview": "The maize leaf exhibits optimal structural integrity, vibrant chlorophyll pigmentation, and no vascular distress.",
            "action": "Immediate Action: None required. The crop is thriving.",
            "chemical": "Chemical Intervention: Withhold all preventative fungicides. Divert farm capital to standard N-P-K nutrient balancing.",
            "organic": "Cultural/Organic Practice: Continue routine field scouting routines twice a week."
        }
    }
    
    return database.get(disease_class, {
        "overview": "Unknown Pathology.",
        "action": "Consult extension officer.", 
        "chemical": "N/A",
        "organic": "N/A"
    })
